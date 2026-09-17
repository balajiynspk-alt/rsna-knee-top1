"""
Unit tests for TGCF-IDS Step 26 Automated Data-Leakage Audit.
"""

import json
from pathlib import Path
import pytest

from src.utils.paths import ProjectPaths
from src.evaluation.leakage_audit import DataLeakageAuditor


def test_leakage_audit_execution_and_artifacts():
    """Verify data leakage auditor runs and produces expected artifacts."""
    auditor = DataLeakageAuditor()
    summary = auditor.run_full_audit()

    assert summary["overall_verdict"] == "PASSED"
    assert summary["total_checks"] == 10
    assert summary["failed_checks"] == 0
    assert summary["passed_checks"] == 10

    json_path = ProjectPaths.RESULTS_FINAL / "leakage_audit.json"
    md_path = ProjectPaths.RESULTS_FINAL / "leakage_audit.md"

    assert json_path.exists(), "leakage_audit.json must exist"
    assert md_path.exists(), "leakage_audit.md must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["overall_verdict"] == "PASSED"
    assert len(data["audit_checklist"]) == 10


def test_individual_leakage_audit_checks():
    """Verify specific leakage checks return PASS status."""
    auditor = DataLeakageAuditor()

    # Preprocessing check
    res2 = auditor.audit_2_preprocessing_leakage()
    assert res2["status"] in ["PASS", "WARNING"]

    # Target leakage check
    res3 = auditor.audit_3_target_leakage()
    assert res3["status"] == "PASS"

    # Train-test graph edges check
    res6 = auditor.audit_6_train_test_graph_edges()
    assert res6["status"] == "PASS"

    # Hyperparameter tuning check
    res9 = auditor.audit_9_test_data_used_during_hyperparameter_tuning()
    assert res9["status"] == "PASS"
