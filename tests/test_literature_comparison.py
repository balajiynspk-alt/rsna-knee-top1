"""
Unit tests for TGCF-IDS Step 27 Literature Comparison & Research Gap artifacts.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.utils.paths import ProjectPaths


def test_literature_comparison_csv_structure_and_contents():
    """Verify literature comparison table exists and contains required schema."""
    lit_csv = ProjectPaths.RESULTS_TABLES / "literature_comparison.csv"
    assert lit_csv.exists(), "literature_comparison.csv must exist"

    df = pd.read_csv(lit_csv)
    assert len(df) >= 10, "Must contain at least 10 reviewed literature works"

    expected_cols = [
        "Year",
        "Model / Paper Name",
        "Dataset",
        "Task",
        "Methodology",
        "Evaluation Protocol",
        "Reported Metrics",
        "Limitations",
        "Relation to TGCF-IDS",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing expected column: {col}"

    # Verify recent publications included (2024, 2025, 2026)
    recent_years = df[df["Year"] >= 2024]
    assert len(recent_years) >= 6, "Must prioritize 2024-2026 literature"


def test_research_gap_markdown_exists_and_populated():
    """Verify research_gap.md exists and contains evidence-based analysis."""
    rg_md = ProjectPaths.RESULTS_TABLES / "research_gap.md"
    assert rg_md.exists(), "research_gap.md must exist"

    content = rg_md.read_text(encoding="utf-8")
    assert len(content) > 1000
    assert "Evidence-Based Research Gap Analysis" in content
    assert "Positioning TGCF-IDS Relative to State-of-the-Art" in content
    assert "Dual-Branch" in content
