import sys
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from kredivo_project.audit import create_audit_sample, evaluate_label_audit


def _imbalanced_reviews():
    rows = []
    # Issue categories with one dominant class and seven smaller categories.
    categories = ["general_satisfaction", "limit_approval", "interest_penalty", "ui_ux",
                  "payment_reconciliation", "login_biometric", "debt_collection_dispute", "app_technical_bug"]
    severities = {"general_satisfaction": "none", "limit_approval": "medium", "interest_penalty": "medium",
                  "ui_ux": "low", "payment_reconciliation": "high", "login_biometric": "high",
                  "debt_collection_dispute": "critical", "app_technical_bug": "medium"}
    counts = [500, 100, 100, 80, 80, 80, 80, 80]
    k = 0
    for category, n in zip(categories, counts):
        for _ in range(n):
            rows.append({"reviewId": f"r{k}", "content": f"review {k}", "score": 3,
                         "issue_category": category, "severity_level": severities[category]})
            k += 1
    return pd.DataFrame(rows)


def test_audit_sample_has_minimum_support_for_each_class(tmp_path):
    sample = create_audit_sample(_imbalanced_reviews(), tmp_path / "audit.csv", 200, 42)
    assert len(sample) == 200
    assert sample.issue_category.value_counts().min() >= 20
    assert sample.severity_level.value_counts().min() >= 10


def test_audit_requires_overall_and_per_class_agreement(tmp_path):
    sample = create_audit_sample(_imbalanced_reviews(), tmp_path / "audit.csv", 200, 42)
    sample["manual_issue_category"] = sample["issue_category"]
    sample["manual_severity_level"] = sample["severity_level"]
    sample["reviewed"] = True
    sample.to_csv(tmp_path / "audit.csv", index=False)
    status = evaluate_label_audit(tmp_path / "audit.csv")
    assert status["passed"]
    assert status["per_class_issue_category"]
    # Introduce severe errors in a minority label; aggregate accuracy must not hide it.
    minority = sample.index[sample.issue_category == "app_technical_bug"]
    sample.loc[minority, "manual_issue_category"] = "general_satisfaction"
    sample.to_csv(tmp_path / "audit.csv", index=False)
    status = evaluate_label_audit(tmp_path / "audit.csv")
    assert not status["passed"]
