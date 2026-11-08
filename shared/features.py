"""Hand-crafted image features for the classical ML baseline.

Every image is reduced to one row of numbers in four groups:

  colour   (col_)  statistics in RGB, HSV and CIELAB, lesion vs skin contrast,
                   hue / saturation / value histograms, the six dermoscopic colours (the C of ABCD)
  texture  (tex_)  the Session 5 trio (Sobel edge density, local-variance roughness,
                   Laplacian variance), LBP histograms and GLCM statistics
  shape    (shp_)  lesion mask from Otsu thresholding: area, compactness, eccentricity, solidity,
                   asymmetry about both principal axes, border irregularity and sharpness
                   (the A, B and D of ABCD)

Metadata features (age, sex, body site) are lesion-level and built in metadata_features().

All functions are pure: an RGB array in [0, 1] goes in, a dict of floats comes out, so they
are easy to test on synthetic images (tests/test_features.py).
"""

import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color, feature, measure, morphology

# 600x450 images are decoded at half scale with JPEG draft mode: 4x fewer pixels, fast,
# and still 300x225 pixels for a lesion that fills most of the frame.
WORK_SIZE = (300, 225)

# Reference colours of the dermoscopic colour criterion (ABCD rule), as sRGB.
DERM_COLOURS = {
    "white": (235, 235, 235),
    "red": (190, 50, 50),
    "light_brown": (190, 130, 90),
    "dark_brown": (100, 60, 40),
    "blue_gray": (110, 125, 150),
    "black": (30, 25, 25),
}
DERM_COLOUR_MAX_DE = 30.0        # a pixel counts for a colour only within this CIELAB distance
DERM_COLOUR_PRESENT = 0.05       # a colour is "present" when it covers 5% of the lesion

SITES = ["head/neck", "lower extremity", "upper extremity", "oral/genital", "unknown"]


# ---------------------------------------------------------------- loading
def load_rgb(path, size=WORK_SIZE):
    """RGB image as float32 in [0, 1], decoded at the working size."""
    with Image.open(path) as img:
        img.draft("RGB", size)
        img = img.convert("RGB")
        if img.size != size:
            img = img.resize(size, Image.BILINEAR)
        return np.asarray(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------- segmentation
def central_ellipse(shape, fraction=0.4):
    """Boolean mask of an ellipse centred in the image, semi-axes = fraction * width / height."""
    h, w = shape
    yy, xx = np.mgrid[:h, :w]
    return ((xx - w / 2) / (fraction * w)) ** 2 + ((yy - h / 2) / (fraction * h)) ** 2 <= 1


def otsu(values, bins=128):
    """Otsu threshold of a 1-D array (between-class variance maximised over a histogram)."""
    counts, edges = np.histogram(values, bins=bins)
    centres = (edges[:-1] + edges[1:]) / 2
    w0 = np.cumsum(counts)
    w1 = w0[-1] - w0
    m0 = np.cumsum(counts * centres) / np.maximum(w0, 1)
    m1 = (np.sum(counts * centres) - np.cumsum(counts * centres)) / np.maximum(w1, 1)
    between = w0 * w1 * (m0 - m1) ** 2
    return centres[int(np.argmax(between))]


def segment_lesion(lab):
    """Lesion mask from the CIELAB lightness, and a flag set when the fallback was used.

    Lesions are darker than the skin around them, so Otsu's threshold on the smoothed L*
    channel (computed inside a central ellipse, which keeps black vignette corners out of the
    histogram) separates lesion from skin. Of the dark components, the one with the most pixels
    inside the central ellipse is kept: MILK10k images are centred on the lesion. Holes are
    filled and the mask smoothed. If nothing plausible is found (under 0.5% or over 95% of the
    frame), a central ellipse covering a quarter of the frame stands in and fallback = 1.
    """
    light = ndimage.gaussian_filter(lab[..., 0], 2)
    centre = central_ellipse(light.shape)
    dark = light < otsu(light[centre])
    dark = ndimage.binary_opening(dark, structure=morphology.disk(3))
    components, n = ndimage.label(dark)
    if n:
        inside = ndimage.sum(centre, components, index=np.arange(1, n + 1))
        best = int(np.argmax(inside)) + 1
        mask = components == best
        mask = np.pad(mask, 5)
        mask = ndimage.binary_closing(mask, structure=morphology.disk(5))[5:-5, 5:-5]
        mask = ndimage.binary_fill_holes(mask)
        share = mask.mean()
        if inside[best - 1] > 0 and 0.005 <= share <= 0.95:
            return mask, 0
    return central_ellipse(light.shape, fraction=0.28), 1


# ---------------------------------------------------------------- colour
def _moments(prefix, values):
    mean, std = float(values.mean()), float(values.std())
    skew = float((((values - mean) / std) ** 3).mean()) if std > 1e-8 else 0.0
    return {f"{prefix}_mean": mean, f"{prefix}_std": std, f"{prefix}_skew": skew}


def colour_features(rgb, lab, hsv, mask):
    f = {}
    channels = {"R": rgb[..., 0], "G": rgb[..., 1], "B": rgb[..., 2],
                "S": hsv[..., 1], "V": hsv[..., 2],
                "L": lab[..., 0], "a": lab[..., 1], "b": lab[..., 2]}
    for name, ch in channels.items():
        f.update(_moments(f"col_{name}", ch.ravel()))

    # hue is circular: a saturation-weighted histogram instead of a mean
    hue_hist, _ = np.histogram(hsv[..., 0], bins=12, range=(0, 1), weights=hsv[..., 1])
    hue_hist = hue_hist / max(hue_hist.sum(), 1e-8)
    f.update({f"col_hue_hist{i:02d}": float(v) for i, v in enumerate(hue_hist)})
    for name, ch in (("S", hsv[..., 1]), ("V", hsv[..., 2])):
        hist, _ = np.histogram(ch, bins=8, range=(0, 1))
        f.update({f"col_{name}_hist{i}": float(v) for i, v in enumerate(hist / ch.size)})

    # lesion against the surrounding skin
    skin = ~ndimage.binary_dilation(mask, structure=morphology.disk(6))
    if skin.sum() < 50:
        skin = ~mask
    for i, name in enumerate("Lab"):
        les, sk = lab[..., i][mask], lab[..., i][skin]
        f[f"col_lesion_{name}_mean"] = float(les.mean())
        f[f"col_lesion_{name}_std"] = float(les.std())
        f[f"col_skin_{name}_mean"] = float(sk.mean()) if sk.size else np.nan
        f[f"col_delta_{name}"] = f[f"col_lesion_{name}_mean"] - f[f"col_skin_{name}_mean"]
    f["col_delta_E"] = float(np.sqrt(sum(f[f"col_delta_{c}"] ** 2 for c in "Lab")))

    # the six dermoscopic colours: share of lesion pixels nearest to each reference
    refs = color.rgb2lab(np.array(list(DERM_COLOURS.values()), dtype=np.float32)[None] / 255.0)[0]
    pixels = lab[mask]
    dist = np.linalg.norm(pixels[:, None, :] - refs[None, :, :], axis=2)
    nearest = dist.argmin(axis=1)
    close = dist.min(axis=1) <= DERM_COLOUR_MAX_DE
    shares = np.bincount(nearest[close], minlength=len(refs)) / max(len(pixels), 1)
    for name, share in zip(DERM_COLOURS, shares):
        f[f"col_derm_{name}"] = float(share)
    f["col_derm_n_colours"] = float((shares >= DERM_COLOUR_PRESENT).sum())
    return f


# ---------------------------------------------------------------- texture
def session5_texture(gray, mask=None):
    """Edge density, roughness and Laplacian variance (Session 5), optionally inside a mask."""
    gx, gy = ndimage.sobel(gray, axis=1), ndimage.sobel(gray, axis=0)
    magnitude = np.hypot(gx, gy)
    local_mean = ndimage.uniform_filter(gray, size=9)
    local_var = ndimage.uniform_filter(gray ** 2, size=9) - local_mean ** 2
    lap = ndimage.laplace(gray)
    sel = np.ones_like(gray, dtype=bool) if mask is None else mask
    return magnitude[sel].mean(), local_var[sel].mean(), lap[sel].var()


def texture_features(gray255, mask):
    f = {}
    for where, sel in (("img", None), ("lesion", mask)):
        edge, rough, lap = session5_texture(gray255, sel)
        f[f"tex_{where}_edge_density"] = float(edge)
        f[f"tex_{where}_roughness"] = float(rough)
        f[f"tex_{where}_laplacian_var"] = float(lap)

    gray8 = np.clip(gray255, 0, 255).astype(np.uint8)
    region = ndimage.binary_dilation(mask, structure=morphology.disk(3))
    for points, radius in ((8, 1), (16, 2)):
        lbp = feature.local_binary_pattern(gray8, points, radius, method="uniform")
        hist, _ = np.histogram(lbp[region], bins=points + 2, range=(0, points + 2))
        hist = hist / max(hist.sum(), 1)
        f.update({f"tex_lbp{points}r{radius}_{i:02d}": float(v) for i, v in enumerate(hist)})

    # GLCM on the lesion's bounding box, 32 grey levels, averaged over four directions
    rows, cols = np.where(mask)
    box = gray8[rows.min():rows.max() + 1, cols.min():cols.max() + 1] // 8
    glcm = feature.graycomatrix(box, distances=[1, 3], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=32, symmetric=True, normed=True)
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation"):
        values = feature.graycoprops(glcm, prop)          # shape (distances, angles)
        for d, dist in enumerate((1, 3)):
            f[f"tex_glcm_{prop}_d{dist}"] = float(np.nan_to_num(values[d].mean()))
    return f


# ---------------------------------------------------------------- shape
def asymmetry(mask):
    """Asymmetry about the two principal axes: share of lesion pixels with no mirror partner.

    Pixel coordinates are centred on the centroid, rotated onto the principal axes and rounded;
    flipping one coordinate mirrors the lesion about the other axis. 0 = perfectly symmetric.
    """
    rows, cols = np.nonzero(mask)
    pts = np.stack([cols - cols.mean(), rows - rows.mean()], axis=1)
    cov = np.cov(pts, rowvar=False)
    _, vecs = np.linalg.eigh(cov)
    uv = np.rint(pts @ vecs[:, ::-1]).astype(np.int64)   # u along the major axis
    offset = np.abs(uv).max() + 1

    def encode(a):
        return (a[:, 0] + offset) * (4 * offset) + (a[:, 1] + offset)

    pixels = np.unique(encode(uv))
    flips = []
    for axis in (1, 0):                                  # mirror about the major, then the minor axis
        mirrored = uv.copy()
        mirrored[:, axis] = -mirrored[:, axis]
        flips.append(1 - np.isin(pixels, np.unique(encode(mirrored))).mean())
    return flips


def shape_features(mask, light, fallback):
    h, w = mask.shape
    props = measure.regionprops(mask.astype(np.uint8))[0]
    area, perimeter = float(props.area), float(max(props.perimeter, 1.0))
    f = {
        "shp_fallback": float(fallback),
        "shp_area_frac": area / (h * w),
        "shp_compactness": float(4 * np.pi * area / perimeter ** 2),
        "shp_eccentricity": float(props.eccentricity),
        "shp_solidity": float(props.solidity),
        "shp_extent": float(props.extent),
        "shp_axis_ratio": float(props.axis_minor_length / max(props.axis_major_length, 1e-8)),
        "shp_centre_offset": float(np.hypot(props.centroid[0] - h / 2, props.centroid[1] - w / 2)
                                   / np.hypot(h, w)),
        "shp_border_touch": float(np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]]).mean()),
    }
    hull = measure.regionprops(props.image_convex.astype(np.uint8))[0]
    f["shp_convexity"] = float(hull.perimeter / perimeter)
    f["shp_asym_major"], f["shp_asym_minor"] = (float(v) for v in asymmetry(mask))

    contour = max(measure.find_contours(np.pad(mask, 1).astype(float), 0.5), key=len)
    radius = np.hypot(contour[:, 0] - 1 - props.centroid[0], contour[:, 1] - 1 - props.centroid[1])
    f["shp_radius_cv"] = float(radius.std() / max(radius.mean(), 1e-8))

    border = mask & ~ndimage.binary_erosion(mask, structure=morphology.disk(2), border_value=1)
    gradient = np.hypot(ndimage.sobel(light, axis=0), ndimage.sobel(light, axis=1))
    f["shp_border_sharpness"] = float(gradient[border].mean()) if border.any() else 0.0
    f["shp_border_sharpness_std"] = float(gradient[border].std()) if border.any() else 0.0
    return f


# ---------------------------------------------------------------- one image
def image_features(rgb):
    """All colour, texture and shape features of one RGB image in [0, 1]. Returns (features, mask)."""
    lab = color.rgb2lab(rgb)
    hsv = color.rgb2hsv(rgb)
    mask, fallback = segment_lesion(lab)
    gray255 = lab[..., 0] * 2.55                         # L* (0-100) on the 0-255 scale of Session 5
    f = {}
    f.update(colour_features(rgb, lab, hsv, mask))
    f.update(texture_features(gray255, mask))
    f.update(shape_features(mask, lab[..., 0], fallback))
    return f, mask


def features_of_file(path):
    features, _ = image_features(load_rgb(path))
    return features


# ---------------------------------------------------------------- metadata
def metadata_features(lesions):
    """Age, sex and body site of each lesion as numbers.

    Age keeps its missing values (NaN): the model pipeline imputes them with the TRAIN median.
    Missing site becomes its own category 'unknown' (Milestone 1, data-quality report).
    Leaky columns (diagnosis_2-4, diagnosis_confirm_type, concomitant_biopsy, melanocytic,
    image_manipulation) are never used.
    """
    out = {"meta_age": lesions["age"].astype(float).to_numpy(),
           "meta_male": (lesions["sex"] == "male").astype(float).to_numpy()}
    site = lesions["site"].fillna("unknown")
    for s in SITES:
        out[f"meta_site_{s.replace('/', '_').replace(' ', '_')}"] = (site == s).astype(float).to_numpy()
    return out
