import pytest
import tempfile
from pathlib import Path
import torch
import numpy as np
from torch_geometric.data import Data

from src.training.train_tgcf_ids import TGCFIDSTrainer
from src.models.tgcf_ids import TGCFIDS


def _create_dummy_graphs(num_graphs=6, nodes_per_graph=30):
    graphs = []
    for g_idx in range(num_graphs):
        x_num = torch.randn(nodes_per_graph, 39)
        x_cat = torch.stack([
            torch.randint(0, 10, (nodes_per_graph,)),
            torch.randint(0, 5, (nodes_per_graph,)),
            torch.randint(0, 3, (nodes_per_graph,)),
        ], dim=-1).float()
        x_dense = torch.randn(nodes_per_graph, 194)

        edge_index = torch.stack([
            torch.randint(0, nodes_per_graph, (nodes_per_graph * 2,)),
            torch.randint(0, nodes_per_graph, (nodes_per_graph * 2,)),
        ], dim=0)
        edge_attr = torch.randn(nodes_per_graph * 2, 6)

        y_multiclass = torch.randint(0, 10, (nodes_per_graph,))
        y_binary = (y_multiclass > 0).long()

        data = Data(
            x_num=x_num,
            x_cat=x_cat,
            x_dense=x_dense,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_multiclass=y_multiclass,
            y_binary=y_binary,
            num_nodes=nodes_per_graph,
        )
        graphs.append(data)
    return graphs


class TestTGCFIDSTrainer:

    def test_trainer_initialization(self):
        trainer = TGCFIDSTrainer(
            load_pretrained=False,
            val_ratio=0.2,
            token_dim=16,
            transformer_heads=2,
            transformer_layers=1,
            graph_hidden_dim=16,
            graph_layers=1,
            fusion_dim=32,
            device="cpu",
            epochs=2,
        )
        assert trainer.model is not None
        assert isinstance(trainer.model, TGCFIDS)
        assert trainer.lr == 1e-3
        assert trainer.epochs == 2

    def test_class_weights_computation(self):
        graphs = _create_dummy_graphs(num_graphs=4, nodes_per_graph=20)
        trainer = TGCFIDSTrainer(
            load_pretrained=False,
            device="cpu",
            token_dim=16,
            transformer_heads=2,
            transformer_layers=1,
            graph_hidden_dim=16,
            graph_layers=1,
            fusion_dim=32,
        )
        weights = trainer._compute_class_weights(graphs)
        assert weights.shape == (10,)
        assert (weights > 0).all()

    def test_train_pipeline_synthetic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            graphs = _create_dummy_graphs(num_graphs=5, nodes_per_graph=20)
            graphs_file = tmp_path / "dummy_train_graphs.pt"
            torch.save(graphs, graphs_file)

            trainer = TGCFIDSTrainer(
                graphs_path=graphs_file,
                load_pretrained=False,
                val_ratio=0.2,
                token_dim=16,
                transformer_heads=2,
                transformer_layers=1,
                graph_hidden_dim=16,
                graph_layers=1,
                fusion_dim=32,
                fusion_strategy="gated",
                batch_size=2,
                epochs=2,
                early_stopping_patience=2,
                use_amp=False,
                device="cpu",
                results_dir=tmp_path / "results",
            )

            res = trainer.train_pipeline()

            assert res["best_epoch"] >= 1
            assert (tmp_path / "results" / "checkpoints" / "best_tgcf_ids.pt").exists()
            assert (tmp_path / "results" / "checkpoints" / "last_tgcf_ids.pt").exists()
            assert (tmp_path / "results" / "tables" / "tgcf_ids_training_history.csv").exists()
            assert (tmp_path / "results" / "tables" / "tgcf_ids_val_metrics.json").exists()
            assert (tmp_path / "results" / "figures" / "tgcf_ids_training_curves.png").exists()

            # Verify history entries
            assert len(trainer.history["epoch"]) == 2
            assert len(trainer.history["train_loss"]) == 2
            assert len(trainer.history["val_macro_f1"]) == 2
            assert all(0.0 <= f1 <= 1.0 for f1 in trainer.history["val_macro_f1"])
            assert all(0.0 <= gate <= 1.0 for gate in trainer.history["mean_gate_val"])
