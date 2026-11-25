from __future__ import annotations
import json
from pathlib import Path
import pandas as pd

AUDIT_COLUMNS = [
    "reviewId", "content", "score", "issue_category", "severity_level",
    "manual_issue_category", "manual_severity_level", "reviewed", "reviewer_notes"
]


def create_audit_sample(
    reviews: pd.DataFrame,
    path: Path,
    sample_size: int = 200,
    random_state: int = 42,
    min_per_issue_category: int = 20,
    min_per_severity: int = 10,
) -> pd.DataFrame:
    """Create a deliberately balanced audit sample.

    A proportional stratified sample of this highly imbalanced dataset can contain
    only one or two examples from rare categories. We therefore first ensure a
    minimum number from every issue category and severity class, then fill to the
    requested size from the remaining population.

    Existing work is preserved once any row has been manually reviewed. A blank
    sample from an older package version is regenerated to improve class coverage.
    """
    if path.exists():
        existing = pd.read_csv(path, keep_default_na=False)
        reviewed = existing.get("reviewed", pd.Series(dtype=str)).astype(str).str.casefold().isin({"true", "1", "yes", "ya"})
        if reviewed.any():
            return existing

    source_cols = [c for c in ["reviewId", "content", "score", "issue_category", "severity_level"] if c in reviews]
    frame = reviews[source_cols].copy()
    if frame.empty:
        raise ValueError("Tidak ada review untuk sampel audit label.")

    rng = __import__("numpy").random.default_rng(int(random_state))
    selected: set[int] = set()

    def enforce_minimum(column: str, minimum: int) -> None:
        if column not in frame:
            return
        for _, group in frame.groupby(column, dropna=False, sort=True):
            already = len(selected.intersection(set(group.index)))
            needed = max(0, min(int(minimum), len(group)) - already)
            if needed:
                available = group.loc[~group.index.isin(selected)]
                if not available.empty:
                    chosen = rng.choice(available.index.to_numpy(), size=min(needed, len(available)), replace=False)
                    selected.update(int(x) for x in chosen)

    enforce_minimum("issue_category", min_per_issue_category)
    enforce_minimum("severity_level", min_per_severity)

    target = min(max(int(sample_size), len(selected)), len(frame))
    remaining = frame.loc[~frame.index.isin(selected)]
    needed = target - len(selected)
    if needed > 0 and not remaining.empty:
        chosen = rng.choice(remaining.index.to_numpy(), size=min(needed, len(remaining)), replace=False)
        selected.update(int(x) for x in chosen)

    sample = frame.loc[sorted(selected)].copy()
    sample = sample.sample(frac=1, random_state=int(random_state))
    sample["manual_issue_category"] = ""
    sample["manual_severity_level"] = ""
    sample["reviewed"] = False
    sample["reviewer_notes"] = ""
    sample = sample[AUDIT_COLUMNS]
    sample.to_csv(path, index=False, encoding="utf-8-sig")
    return sample


def _class_agreement(frame: pd.DataFrame, reference_col: str, manual_col: str) -> dict:
    details = {}
    for label, group in frame.groupby(reference_col, dropna=False, sort=True):
        label_text = str(label)
        agreement = float((group[reference_col].astype(str).str.strip() == group[manual_col].astype(str).str.strip()).mean())
        details[label_text] = {"n": int(len(group)), "agreement": agreement}
    return details


def evaluate_label_audit(
    path: Path,
    min_agreement: float = 0.80,
    min_class_agreement: float = 0.70,
    min_class_support: int = 10,
) -> dict:
    status = {
        "status": "not_audited",
        "passed": False,
        "sample_size": 0,
        "reviewed_count": 0,
        "agreement_issue_category": None,
        "agreement_severity_level": None,
        "macro_agreement_issue_category": None,
        "macro_agreement_severity_level": None,
        "per_class_issue_category": {},
        "per_class_severity_level": {},
        "minimum_agreement": float(min_agreement),
        "minimum_class_agreement": float(min_class_agreement),
        "minimum_class_support": int(min_class_support),
        "reason": "Sampel audit belum diisi oleh peninjau manusia."
    }
    if not path.exists():
        return status
    df = pd.read_csv(path, keep_default_na=False)
    status["sample_size"] = int(len(df))
    if df.empty or "reviewed" not in df.columns:
        status["reason"] = "File audit kosong atau kolom reviewed tidak ditemukan."
        return status
    reviewed = df["reviewed"].astype(str).str.casefold().isin({"true", "1", "yes", "ya"})
    status["reviewed_count"] = int(reviewed.sum())
    required = ["issue_category", "severity_level", "manual_issue_category", "manual_severity_level"]
    if any(c not in df.columns for c in required):
        status["reason"] = "Kolom label sistem atau label manual belum lengkap."
        return status
    complete = (
        reviewed
        & df["manual_issue_category"].astype(str).str.strip().ne("")
        & df["manual_severity_level"].astype(str).str.strip().ne("")
    )
    if not bool(complete.all()):
        status["reason"] = "Belum semua sampel memiliki label manual dan status reviewed=true."
        return status

    checked = df.loc[complete].copy()
    checked["issue_category"] = checked["issue_category"].astype(str).str.strip()
    checked["manual_issue_category"] = checked["manual_issue_category"].astype(str).str.strip()
    checked["severity_level"] = checked["severity_level"].astype(str).str.strip()
    checked["manual_severity_level"] = checked["manual_severity_level"].astype(str).str.strip()

    issue_agree = float((checked["issue_category"] == checked["manual_issue_category"]).mean())
    severity_agree = float((checked["severity_level"] == checked["manual_severity_level"]).mean())
    issue_details = _class_agreement(checked, "issue_category", "manual_issue_category")
    severity_details = _class_agreement(checked, "severity_level", "manual_severity_level")
    issue_macro = float(sum(x["agreement"] for x in issue_details.values()) / max(1, len(issue_details)))
    severity_macro = float(sum(x["agreement"] for x in severity_details.values()) / max(1, len(severity_details)))

    under_supported = [
        f"issue_category:{label}" for label, item in issue_details.items() if item["n"] < min_class_support
    ] + [
        f"severity_level:{label}" for label, item in severity_details.items() if item["n"] < min_class_support
    ]
    weak_classes = [
        f"issue_category:{label}" for label, item in issue_details.items() if item["agreement"] < min_class_agreement
    ] + [
        f"severity_level:{label}" for label, item in severity_details.items() if item["agreement"] < min_class_agreement
    ]

    passed = (
        issue_agree >= min_agreement
        and severity_agree >= min_agreement
        and issue_macro >= min_class_agreement
        and severity_macro >= min_class_agreement
        and not under_supported
        and not weak_classes
    )
    reasons = []
    if issue_agree < min_agreement:
        reasons.append("agreement issue_category di bawah ambang")
    if severity_agree < min_agreement:
        reasons.append("agreement severity_level di bawah ambang")
    if issue_macro < min_class_agreement or severity_macro < min_class_agreement:
        reasons.append("macro agreement per kelas di bawah ambang")
    if under_supported:
        reasons.append("kelas belum memiliki sampel minimum: " + ", ".join(under_supported))
    if weak_classes:
        reasons.append("kelas dengan agreement rendah: " + ", ".join(weak_classes))

    status.update({
        "agreement_issue_category": issue_agree,
        "agreement_severity_level": severity_agree,
        "macro_agreement_issue_category": issue_macro,
        "macro_agreement_severity_level": severity_macro,
        "per_class_issue_category": issue_details,
        "per_class_severity_level": severity_details,
        "passed": bool(passed),
        "status": "passed" if passed else "needs_revision",
        "reason": "Audit memenuhi ambang keseluruhan dan per kelas." if passed else "; ".join(reasons),
    })
    return status


def save_audit_status(path: Path, status: dict) -> None:
    path.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
