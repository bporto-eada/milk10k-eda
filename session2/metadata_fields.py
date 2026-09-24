import math

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

from shared.paths import LABEL
from figstyle import colour, tidy, finish

IGNORE = {"isic_id", "lesion_id", LABEL}
MIN_GROUP = 20
MAX_BARS = 12

WARNINGS = {
    "diagnosis_2": "leakage, it is a finer version of the target",
    "diagnosis_3": "leakage, it is a finer version of the target",
    "diagnosis_4": "leakage, it is a finer version of the target",
    "subclass": "leakage, the 11 class label",
    "melanocytic": "leakage, follows from the diagnosis",
    "diagnosis_confirm_type": "leakage, encodes how suspicious the doctor was",
    "concomitant_biopsy": "leakage, suspicious lesions get biopsied",
    "attribution": "bias, identifies the clinic",
    "copyright_license": "bias, identifies the clinic",
    "image_manipulation": "bias, identifies the device or processing",
    "image_type": "confounder, changes colours but not the diagnosis",
}


def field_kind(series):
    values = series.dropna()
    if values.empty:
        return "empty"
    lowered = set(values.astype(str).str.lower().unique())
    if pd.api.types.is_bool_dtype(values) or lowered <= {"true", "false"}:
        return "boolean"
    if pd.api.types.is_numeric_dtype(values):
        return "numeric"
    if values.astype(str).str.len().mean() > 40:
        return "text"
    return "categorical"


def lesion_level(df, col):
    varies = (df.groupby("lesion_id")[col].nunique(dropna=False) > 1).any()
    return df if varies else df.drop_duplicates("lesion_id")


def field_summary(df):
    rows = []
    for col in df.columns:
        if col in IGNORE:
            continue
        rows.append({
            "field": col,
            "kind": field_kind(df[col]),
            "distinct": df[col].nunique(),
            "missing_pct": round(df[col].isna().mean() * 100, 1),
            "warning": WARNINGS.get(col, ""),
        })
    return pd.DataFrame(rows)


def categorical_link(df, col):
    d = lesion_level(df, col)
    x = d[col].astype(str)
    frequency = x.map(x.value_counts())
    x = x.where(frequency >= MIN_GROUP, "(other)")
    table = pd.crosstab(x, d[LABEL])
    if len(table) < 2:
        return None
    chi2, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = math.sqrt(chi2 / (n * (min(table.shape) - 1)))
    return {"field": col, "test": "chi-square", "strength": v, "p": p, "table": table}


def numeric_link(df, col):
    d = lesion_level(df, col).dropna(subset=[col])
    groups = [g[col].to_numpy() for _, g in d.groupby(LABEL)]
    h, p = stats.kruskal(*groups)
    k, n = len(groups), len(d)
    eta = max((h - k + 1) / (n - k), 0.0)
    return {"field": col, "test": "Kruskal-Wallis", "strength": eta, "p": p}


def chart_categorical(result):
    table = result["table"]
    order = table.sum(axis=1).sort_values(ascending=False).index[:MAX_BARS]
    share = table.loc[order].div(table.loc[order].sum(axis=1), axis=0) * 100

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(share) + 1.5))
    left = pd.Series(0.0, index=share.index)
    for cls in share.columns:
        ax.barh(share.index.astype(str), share[cls], left=left, height=0.65,
                color=colour(cls), edgecolor="white", label=cls)
        left = left + share[cls]
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of lesions")
    ax.set_title(f"{result['field']} vs {LABEL}   Cramer's V = {result['strength']:.2f}", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    finish(fig, f"s2_meta_{result['field']}.png")


def chart_numeric(df, col, result):
    d = lesion_level(df, col).dropna(subset=[col])
    classes = list(d[LABEL].value_counts().index)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))

    box = left.boxplot([d.loc[d[LABEL] == c, col] for c in classes], patch_artist=True)
    for patch, c in zip(box["boxes"], classes):
        patch.set_facecolor(colour(c))
        patch.set_alpha(0.55)
    left.set_xticks(range(1, len(classes) + 1), classes)
    left.set_ylabel(col)
    tidy(left)

    for c in classes:
        right.hist(d.loc[d[LABEL] == c, col], bins=17, density=True, histtype="stepfilled",
                   alpha=0.35, color=colour(c), edgecolor=colour(c), linewidth=1.5, label=c)
    right.set_xlabel(col)
    right.legend(frameon=False)
    tidy(right)

    fig.suptitle(f"{col} by class   Kruskal-Wallis p = {result['p']:.1e}, "
                 f"eta squared = {result['strength']:.3f}", fontsize=10)
    finish(fig, f"s2_meta_{col}.png")


def chart_overview(results):
    r = pd.DataFrame(results).sort_values("strength")
    colours = ["#c8553d" if f in WARNINGS else "#3b6ea5" for f in r["field"]]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(r) + 1.5))
    ax.hlines(r["field"], 0, r["strength"], color=colours, linewidth=2.5)
    ax.scatter(r["strength"], r["field"], color=colours, s=60, zorder=3)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel(f"strength of association with {LABEL} (0 none, 1 perfect)")
    ax.set_title("Which metadata fields predict the diagnosis, red = do not use as a feature", fontsize=10)
    ax.xaxis.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    finish(fig, "s2_meta_overview.png")


def run(df, say):
    summary = field_summary(df)
    say(summary.to_string(index=False))

    results = []
    for _, row in summary.iterrows():
        col = row["field"]
        if row["kind"] in ("categorical", "boolean"):
            result = categorical_link(df, col)
            if result:
                chart_categorical(result)
                results.append(result)
        elif row["kind"] == "numeric":
            result = numeric_link(df, col)
            chart_numeric(df, col, result)
            results.append(result)

    chart_overview(results)
    ranking = pd.DataFrame([{k: v for k, v in r.items() if k != "table"} for r in results])
    ranking["warning"] = ranking["field"].map(WARNINGS).fillna("")
    ranking = ranking.sort_values("strength", ascending=False)
    say(f"\nStrength of association with {LABEL}, strongest first:")
    say(ranking.round(3).to_string(index=False))
    return ranking
