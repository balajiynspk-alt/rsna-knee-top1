import os
import unittest
import pandas as pd


class TestPaperPackage(unittest.TestCase):
    def setUp(self):
        self.package_dir = os.path.join(os.path.dirname(__file__), "..", "results", "paper_package")
        self.tables_dir = os.path.join(self.package_dir, "tables")
        self.figures_dir = os.path.join(self.package_dir, "figures")
        self.compendium_path = os.path.join(self.package_dir, "paper_results_compendium.md")

    def test_tables_exist_and_not_empty(self):
        expected_tables = [
            "table1_dataset_statistics.csv",
            "table2_traditional_ml_baselines.csv",
            "table3_deep_learning_baselines.csv",
            "table4_gnn_transformer_baselines.csv",
            "table5_ablation.csv",
            "table6_per_class_performance.csv",
            "table7_cross_dataset.csv",
            "table8_robustness.csv",
            "table9_efficiency.csv",
            "table10_reproducibility.csv"
        ]
        for tbl in expected_tables:
            tbl_path = os.path.join(self.tables_dir, tbl)
            self.assertTrue(os.path.exists(tbl_path), f"Missing table: {tbl}")
            df = pd.read_csv(tbl_path)
            self.assertGreater(len(df), 0, f"Table {tbl} is empty")

    def test_figures_exist_and_non_zero_size(self):
        expected_figures = [
            "figure1_tgcf_ids_architecture.png",
            "figure2_dataset_class_distribution.png",
            "figure3_graph_example.png",
            "figure4_training_curves.png",
            "figure5_confusion_matrix.png",
            "figure6_per_class_f1.png",
            "figure7_ablation_study.png",
            "figure8_cross_dataset_performance.png",
            "figure9_robustness_curves.png",
            "figure10_explainability.png",
            "figure11_pareto_cost_vs_performance.png"
        ]
        for fig in expected_figures:
            fig_path = os.path.join(self.figures_dir, fig)
            self.assertTrue(os.path.exists(fig_path), f"Missing figure: {fig}")
            self.assertGreater(os.path.getsize(fig_path), 1000, f"Figure {fig} has abnormally small size")

    def test_compendium_markdown_exists(self):
        self.assertTrue(os.path.exists(self.compendium_path), "Compendium markdown file missing")
        with open(self.compendium_path, "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("TABLE 1: Dataset Statistics", content)
        self.assertIn("TABLE 10: Multi-Seed Reproducibility", content)
        self.assertIn("FIGURE 1:", content)
        self.assertIn("FIGURE 11:", content)


if __name__ == "__main__":
    unittest.main()
