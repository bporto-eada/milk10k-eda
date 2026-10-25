# Lesion segmentation check

Otsu threshold on smoothed CIELAB L*, the dark component with most pixels in the central ellipse (shared/features.py, segment_lesion). The fallback is a central ellipse, used when no plausible component was found.

| image_type | diagnosis_1 | images | fallback_pct | median_lesion_area_pct |
|---|---|---|---|---|
| clinical: close-up | Benign | 1483 | 1.3 | 21.4 |
| clinical: close-up | Indeterminate | 123 | 1.6 | 35.8 |
| clinical: close-up | Malignant | 3634 | 0.7 | 38.9 |
| dermoscopic | Benign | 1483 | 2.3 | 23.1 |
| dermoscopic | Indeterminate | 123 | 1.6 | 27.8 |
| dermoscopic | Malignant | 3634 | 5.4 | 25.0 |
