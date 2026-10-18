"""
Per-feature data checks for the Women's E-Commerce Clothing Reviews dataset.

Pure functions: DataFrame in, metrics out. Nothing here reads files, writes files or
modifies the input, so every check can be unit-tested and reused (training-time audit,
CI gate, and later a drift check on fresh data).

Each check records:
  * metrics -> numbers for the report (JSON-serialisable)
  * issues  -> findings with a severity:
        "error"   = breaks the data contract, the audit stage should fail
        "warning" = worth a human look, does not stop the pipeline
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- config
@dataclass
class AuditConfig:
    # column names as they appear in the raw CSV (index artifact "Unnamed: 0" is dropped first)
    col_id: str = "Clothing ID"
    col_age: str = "Age"
    col_title: str = "Title"
    col_text: str = "Review Text"
    col_rating: str = "Rating"
    col_rec: str = "Recommended IND"
    col_pfc: str = "Positive Feedback Count"
    col_division: str = "Division Name"
    col_department: str = "Department Name"
    col_class: str = "Class Name"

    # thresholds: starting points, tune after reading your first real report
    age_min: int = 13
    age_max: int = 100
    short_review_words: int = 3
    long_review_chars: int = 2000
    min_reviews_per_product: int = 3
    fuzzy_cutoff: float = 0.8

    # data-contract gates (percent of rows unless stated)
    max_invalid_rating_pct: float = 0.5
    max_invalid_rec_pct: float = 0.5
    max_invalid_age_pct: float = 1.0
    max_null_pct: dict = field(default_factory=lambda: {
        "Clothing ID": 0.0, "Rating": 0.0, "Recommended IND": 0.0,
        "Review Text": 10.0, "Division Name": 1.0, "Department Name": 1.0, "Class Name": 1.0,
    })
    risk_rate_bounds: tuple = (0.05, 0.60)   # proxy-target sanity range

    # look-alike pairs that are legitimately different values (suppressed in the cross-column check)
    ignore_lookalikes: tuple = (("Intimate", "Intimates"),)

    @property
    def expected_cols(self) -> list[str]:
        return [self.col_id, self.col_age, self.col_title, self.col_text, self.col_rating,
                self.col_rec, self.col_pfc, self.col_division, self.col_department, self.col_class]

    @property
    def cat_cols(self) -> list[str]:
        return [self.col_division, self.col_department, self.col_class]


class Audit:
    """Collects metrics and issues across all checks."""

    def __init__(self) -> None:
        self.metrics: dict = {}
        self.issues: list[dict] = []

    def add_issue(self, feature: str, severity: str, message: str) -> None:
        assert severity in ("error", "warning")
        self.issues.append({"feature": feature, "severity": severity, "message": message})

    @property
    def errors(self) -> list[dict]:
        return [i for i in self.issues if i["severity"] == "error"]

    @property
    def warnings(self) -> list[dict]:
        return [i for i in self.issues if i["severity"] == "warning"]


# --------------------------------------------------------------------------- helpers
def _py(obj):
    """Make numpy/pandas scalars and containers JSON-serialisable."""
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Series):
        return _py(obj.to_dict())
    return obj


def _pct(part, whole) -> float:
    return round(100.0 * float(part) / float(whole), 3) if whole else 0.0


def drop_index_artifacts(df: pd.DataFrame) -> pd.DataFrame:
    """Remove CSV-export index columns such as 'Unnamed: 0'."""
    return df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])


# --------------------------------------------------------------------------- checks
def check_schema(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> bool:
    missing = [c for c in cfg.expected_cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cfg.expected_cols]
    audit.metrics["schema"] = {"rows": len(df), "columns": list(df.columns),
                               "missing_columns": missing, "unexpected_columns": extra}
    if missing:
        audit.add_issue("schema", "error", f"Missing expected columns: {missing}")
    if extra:
        audit.add_issue("schema", "warning", f"Unexpected extra columns: {extra}")
    return not missing


def check_overview(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    n = len(df)
    nulls = df.isna().sum()
    audit.metrics["overview"] = {
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "null_count": nulls.to_dict(),
        "null_pct": {c: _pct(v, n) for c, v in nulls.items()},
        "fully_duplicated_rows": int(df.duplicated().sum()),
    }
    if audit.metrics["overview"]["fully_duplicated_rows"]:
        audit.add_issue("overview", "warning",
                        f"{audit.metrics['overview']['fully_duplicated_rows']} fully duplicated rows")
    for col, limit in cfg.max_null_pct.items():
        if col in df.columns:
            pct = _pct(nulls[col], n)
            if pct > limit:
                audit.add_issue(col, "error", f"{pct}% nulls exceeds the allowed {limit}%")


def check_clothing_id(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    s = df[cfg.col_id]
    per_product = df.groupby(cfg.col_id).size()
    thin = per_product < cfg.min_reviews_per_product
    audit.metrics[cfg.col_id] = {
        "dtype": str(s.dtype), "nulls": int(s.isna().sum()), "unique_products": int(s.nunique()),
        "reviews_per_product": per_product.describe().round(2).to_dict(),
        "products_below_min_reviews": int(thin.sum()),
        "products_below_min_reviews_pct": _pct(thin.sum(), len(per_product)),
        "reviews_in_thin_products_pct": _pct(per_product[thin].sum(), len(df)),
    }


def check_age(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    raw = df[cfg.col_age]
    age = pd.to_numeric(raw, errors="coerce")
    non_numeric = int(age.isna().sum() - raw.isna().sum())
    invalid = age.notna() & ~age.between(cfg.age_min, cfg.age_max)
    q1, q3 = age.quantile([0.25, 0.75])
    iqr = q3 - q1
    audit.metrics[cfg.col_age] = {
        "nulls": int(raw.isna().sum()), "non_numeric": non_numeric,
        "describe": age.describe().round(2).to_dict(),
        "percentiles": age.quantile([0.01, 0.05, 0.5, 0.95, 0.99]).round(2).to_dict(),
        "invalid_count": int(invalid.sum()),
        "iqr_outliers_informational": int(((age < q1 - 1.5 * iqr) | (age > q3 + 1.5 * iqr)).sum()),
    }
    if non_numeric:
        audit.add_issue(cfg.col_age, "warning", f"{non_numeric} non-numeric values will become NaN")
    pct = _pct(invalid.sum(), len(df))
    if invalid.sum():
        audit.add_issue(cfg.col_age, "error" if pct > cfg.max_invalid_age_pct else "warning",
                        f"{int(invalid.sum())} ages outside {cfg.age_min}-{cfg.age_max} ({pct}%)")


def check_title(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    s = df[cfg.col_title]
    t = s.dropna().astype(str)
    audit.metrics[cfg.col_title] = {
        "missing": int(s.isna().sum()), "missing_pct": _pct(s.isna().sum(), len(df)),
        "empty_or_whitespace": int((t.str.strip() == "").sum()),
        "no_letters": int((~t.str.contains(r"[A-Za-z]")).sum()),
        "duplicated_titles": int(t.duplicated().sum()),
        "length": t.str.len().describe().round(2).to_dict(),
        "top_titles": t.value_counts().head(5).to_dict(),
    }


def check_review_text(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    s = df[cfg.col_text]
    stripped = s.dropna().astype(str).str.strip()
    chars = stripped.str.len()
    words = stripped.str.split().str.len()
    with_text = df.dropna(subset=[cfg.col_text])
    same_product = int(with_text.duplicated(subset=[cfg.col_id, cfg.col_text]).sum())
    anywhere = int(with_text.duplicated(subset=[cfg.col_text]).sum())
    non_ascii = stripped.apply(lambda x: any(ord(ch) > 127 for ch in x)).mean() if len(stripped) else 0.0
    audit.metrics[cfg.col_text] = {
        "missing": int(s.isna().sum()), "missing_pct": _pct(s.isna().sum(), len(df)),
        "whitespace_only": int((stripped == "").sum()),
        "chars": chars.describe().round(2).to_dict(),
        "short_reviews": int((words < cfg.short_review_words).sum()),
        "long_reviews": int((chars > cfg.long_review_chars).sum()),
        "dupes_same_product_same_text": same_product,
        "dupes_different_product_same_text": anywhere - same_product,
        "non_ascii_pct": round(100 * float(non_ascii), 3),
        "contains_retailer_token_pct": round(
            100 * float(stripped.str.lower().str.contains("retailer").mean()) if len(stripped) else 0.0, 3),
    }
    if same_product:
        audit.add_issue(cfg.col_text, "warning",
                        f"{same_product} rows repeat the same text on the same product (likely double-posts)")


def check_rating(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    raw = df[cfg.col_rating]
    r = pd.to_numeric(raw, errors="coerce")
    invalid = r.notna() & ~r.isin([1, 2, 3, 4, 5])
    freq = r.value_counts().sort_index()
    audit.metrics[cfg.col_rating] = {
        "nulls": int(raw.isna().sum()), "non_numeric": int(r.isna().sum() - raw.isna().sum()),
        "invalid_count": int(invalid.sum()),
        "frequencies": freq.to_dict(),
        "shares": (freq / freq.sum()).round(3).to_dict() if freq.sum() else {},
    }
    pct = _pct(invalid.sum(), len(df))
    if invalid.sum():
        audit.add_issue(cfg.col_rating, "error" if pct > cfg.max_invalid_rating_pct else "warning",
                        f"{int(invalid.sum())} ratings outside 1-5 ({pct}%)")


def check_recommended(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    raw = df[cfg.col_rec]
    rec = pd.to_numeric(raw, errors="coerce")
    r = pd.to_numeric(df[cfg.col_rating], errors="coerce")
    invalid = rec.notna() & ~rec.isin([0, 1])
    audit.metrics[cfg.col_rec] = {
        "nulls": int(raw.isna().sum()), "unique_values": sorted(rec.dropna().unique().tolist()),
        "invalid_count": int(invalid.sum()),
        "recommend_rate_pct": round(100 * float(rec.mean()), 2) if rec.notna().any() else None,
        "contradiction_high_rating_not_recommended": int(((r >= 4) & (rec == 0)).sum()),
        "contradiction_low_rating_recommended": int(((r <= 2) & (rec == 1)).sum()),
    }
    pct = _pct(invalid.sum(), len(df))
    if invalid.sum():
        audit.add_issue(cfg.col_rec, "error" if pct > cfg.max_invalid_rec_pct else "warning",
                        f"{int(invalid.sum())} values are not 0/1 ({pct}%)")
    for key in ("contradiction_high_rating_not_recommended", "contradiction_low_rating_recommended"):
        n = audit.metrics[cfg.col_rec][key]
        if n:
            audit.add_issue(cfg.col_rec, "warning",
                            f"{n} rows: {key.replace('contradiction_', '').replace('_', ' ')} (label noise in the proxy target)")


def check_positive_feedback(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    raw = df[cfg.col_pfc]
    p = pd.to_numeric(raw, errors="coerce")
    audit.metrics[cfg.col_pfc] = {
        "nulls": int(raw.isna().sum()), "negative_values": int((p < 0).sum()),
        "describe": p.describe().round(2).to_dict(),
        "percentiles": p.quantile([0.5, 0.75, 0.9, 0.95, 0.99, 0.999]).round(2).to_dict(),
        "zero_pct": round(100 * float((p == 0).mean()), 2),
        "skew": round(float(p.skew()), 3),
        "top_values": p.nlargest(5).tolist(),
    }
    if (p < 0).sum():
        audit.add_issue(cfg.col_pfc, "error", f"{int((p < 0).sum())} negative counts")


def check_category(df: pd.DataFrame, col: str, cfg: AuditConfig, audit: Audit) -> None:
    s = df[col]
    nn = s.dropna().astype(str)
    vc = nn.value_counts()
    ws = int((nn != nn.str.strip()).sum())
    unique_raw = int(nn.nunique())
    unique_norm = int(nn.str.strip().str.lower().nunique())
    cats = list(vc.index)
    pairs = []
    for i, a in enumerate(cats):
        for b in cats[i + 1:]:
            ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
            if ratio >= cfg.fuzzy_cutoff:
                pairs.append([a, b, round(ratio, 2)])
    audit.metrics[col] = {
        "nulls": int(s.isna().sum()), "null_pct": _pct(s.isna().sum(), len(df)),
        "unique": unique_raw, "unique_after_strip_lower": unique_norm,
        "counts": vc.to_dict(), "shares": (vc / vc.sum()).round(4).to_dict() if vc.sum() else {},
        "whitespace_variants": ws, "near_duplicate_pairs": pairs,
        "top_category_share_pct": round(100 * float(vc.iloc[0] / vc.sum()), 2) if len(vc) else None,
        "smallest_category_count": int(vc.iloc[-1]) if len(vc) else None,
    }
    if ws:
        audit.add_issue(col, "warning", f"{ws} values have leading/trailing whitespace")
    if unique_norm < unique_raw:
        audit.add_issue(col, "warning", "case/whitespace variants of the same category exist")
    if pairs:
        audit.add_issue(col, "warning", f"near-duplicate spellings inside the column: {pairs}")


def check_cross_column_lookalikes(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    """A typo in one column can look like a legitimate value from another (e.g. 'Initmates' vs 'Intimates')."""
    ignore = {frozenset(p) for p in cfg.ignore_lookalikes}
    vocab = {c: sorted(df[c].dropna().astype(str).str.strip().unique()) for c in cfg.cat_cols}
    found, seen = [], set()
    for c in cfg.cat_cols:
        for other in cfg.cat_cols:
            if other == c:
                continue
            for v in vocab[c]:
                if v in vocab[other]:
                    continue
                m = difflib.get_close_matches(v, vocab[other], n=1, cutoff=cfg.fuzzy_cutoff)
                if not m or frozenset((v, m[0])) in ignore:
                    continue
                key = frozenset({(c, v), (other, m[0])})      # report each pair once, not once per direction
                if key in seen:
                    continue
                seen.add(key)
                found.append({"column": c, "value": v, "looks_like": m[0], "in_column": other})
    audit.metrics["cross_column_lookalikes"] = found
    for f in found:
        audit.add_issue(f["column"], "warning",
                        f"'{f['value']}' looks like '{f['looks_like']}' from {f['in_column']} (possible misspelling)")


def check_hierarchy(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    sub = df.dropna(subset=cfg.cat_cols)
    cls_dept = sub.groupby(cfg.col_class)[cfg.col_department].nunique()
    dept_div = sub.groupby(cfg.col_department)[cfg.col_division].nunique()
    null_pattern = df[cfg.cat_cols].isna().value_counts()
    audit.metrics["hierarchy"] = {
        "classes_under_multiple_departments": cls_dept[cls_dept > 1].to_dict(),
        "departments_under_multiple_divisions": dept_div[dept_div > 1].to_dict(),
        "null_patterns": {"|".join(map(str, k)): int(v) for k, v in null_pattern.items()},
    }
    if len(cls_dept[cls_dept > 1]):
        audit.add_issue("hierarchy", "warning",
                        f"classes under more than one department: {cls_dept[cls_dept > 1].to_dict()}")


def check_target(df: pd.DataFrame, cfg: AuditConfig, audit: Audit) -> None:
    r = pd.to_numeric(df[cfg.col_rating], errors="coerce")
    rec = pd.to_numeric(df[cfg.col_rec], errors="coerce")
    valid = r.notna() & rec.notna()
    risk = ((r <= 2) | (rec == 0)).astype(int)[valid]
    rate = float(risk.mean()) if len(risk) else float("nan")
    audit.metrics["proxy_target"] = {
        "risk_rate": round(rate, 4) if not np.isnan(rate) else None,
        "low_rating_only": int(((r <= 2) & (rec == 1)).sum()),
        "not_recommended_only": int(((r > 2) & (rec == 0)).sum()),
        "both": int(((r <= 2) & (rec == 0)).sum()),
    }
    lo, hi = cfg.risk_rate_bounds
    if np.isnan(rate) or not (lo <= rate <= hi):
        audit.add_issue("proxy_target", "warning", f"risk rate {rate:.1%} is outside the sane range {lo:.0%}-{hi:.0%}")


# --------------------------------------------------------------------------- orchestration
def run_all_checks(raw: pd.DataFrame, cfg: AuditConfig | None = None) -> Audit:
    """Run every check on a raw frame. Never mutates `raw`."""
    cfg = cfg or AuditConfig()
    audit = Audit()
    df = drop_index_artifacts(raw)

    if not check_schema(df, cfg, audit):
        audit.metrics = _py(audit.metrics)
        return audit            # the remaining checks assume the schema is intact

    check_overview(df, cfg, audit)
    check_clothing_id(df, cfg, audit)
    check_age(df, cfg, audit)
    check_title(df, cfg, audit)
    check_review_text(df, cfg, audit)
    check_rating(df, cfg, audit)
    check_recommended(df, cfg, audit)
    check_positive_feedback(df, cfg, audit)
    for col in cfg.cat_cols:
        check_category(df, col, cfg, audit)
    check_cross_column_lookalikes(df, cfg, audit)
    check_hierarchy(df, cfg, audit)
    check_target(df, cfg, audit)

    audit.metrics = _py(audit.metrics)
    return audit