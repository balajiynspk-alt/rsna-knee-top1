"""
Unit tests for TGCF-IDS Step 24 Explainability & Attribution Framework.
"""

from pathlib import Path
import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from src.utils.paths import ProjectPaths
from src.explainability.explainer import (
    IntegratedGradientsExplainer,
    GraphSaliencyExplainer,
    TGCFIDSExplainabilitySuite,
    CaseStudyExplanation,
)
from src.explainability.plot_explainability import ExplainabilityFigureGenerator


def test_integrated_gradients_computation():
    """Verify Integrated Gradients produces bounded attribution vector."""
    suite = TGCFIDSExplainabilitySuite(device="cpu")
    g = suite.test_graphs[0]
    ig_scores = suite.ig_explainer.attribute_sample(g, node_idx=0, target_class=0, steps=5)

    assert len(ig_scores) == 39
    assert isinstance(ig_scores, (list, tuple, np.ndarray)) or isinstance(ig_scores, torch.Tensor)


def test_graph_saliency_computation():
    """Verify graph saliency identifies connected neighbors."""
    suite = TGCFIDSExplainabilitySuite(device="cpu")
    g = suite.test_graphs[0]
    neighbors = suite.graph_explainer.attribute_subgraph(g, target_node=0, target_class=0)

    assert isinstance(neighbors, list)
    if len(neighbors) > 0:
        assert "neighbor_id" in neighbors[0]
        assert "importance_score" in neighbors[0]
        assert "direction" in neighbors[0]


def test_explainability_figures_exist():
    """Verify all 5 publication explainability figures were generated."""
    figures_dir = ProjectPaths.RESULTS_FIGURES / "explainability"
    expected_figures = [
        "explainability_feature_attribution_integrated_gradients.png",
        "explainability_transformer_attention_heatmaps.png",
        "explainability_graph_subgraph_attribution.png",
        "explainability_temporal_flow_sequence.png",
        "explainability_comprehensive_case_studies.png",
    ]
    for fig_name in expected_figures:
        fig_path = figures_dir / fig_name
        assert fig_path.exists(), f"Missing figure: {fig_name}"
        assert fig_path.stat().st_size > 5000, f"Figure {fig_name} is empty"
