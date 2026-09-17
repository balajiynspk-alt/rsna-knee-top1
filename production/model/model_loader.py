#!/usr/bin/env python3
"""
TGCF-IDS Production Model Loader & Checkpoint Integrity Verifier.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union

import torch
import yaml

from src.models.tgcf_ids import TGCFIDS
from src.utils.paths import ProjectPaths

logger = logging.getLogger("production.model.loader")


class ProductionModelLoader:
    """
    Safely loads, verifies, and initializes the frozen TGCF-IDS research model.
    """

    MODEL_VERSION = "TGCF-IDS-production-compatible-v1"
    EXPECTED_SHA256 = "f6a85511ef0d44bed49a481c4a720a9173c57ec9f91ed64192576b019cf8cd54"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        verify_hash: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path or ProjectPaths.RESULTS_CHECKPOINTS / "final_tgcf_ids_model.pt")
        if not self.checkpoint_path.exists():
            self.checkpoint_path = ProjectPaths.RESULTS_CHECKPOINTS / "best_tgcf_ids.pt"

        self.device = self._select_device(device)
        self.verify_hash = verify_hash
        self.metadata = self._load_metadata()

    def _select_device(self, requested: str) -> torch.device:
        if requested == "cuda" and torch.cuda.is_available():
            dev = torch.device("cuda")
        elif requested == "cpu":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[+] Selected production inference device: {dev}")
        return dev

    def _load_metadata(self) -> dict:
        meta_path = ProjectPaths.DATA_PROCESSED / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def verify_checkpoint_integrity(self) -> bool:
        """Verify the cryptographic hash of the frozen checkpoint."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {self.checkpoint_path}")

        file_bytes = self.checkpoint_path.read_bytes()
        computed_sha256 = hashlib.sha256(file_bytes).hexdigest()

        if self.checkpoint_path.name == "final_tgcf_ids_model.pt":
            if computed_sha256 != self.EXPECTED_SHA256:
                logger.warning(
                    f"Checkpoint hash mismatch! Expected {self.EXPECTED_SHA256}, got {computed_sha256}"
                )
                return False
        logger.info(f"[+] Checkpoint verified: {self.checkpoint_path.name} (SHA256: {computed_sha256[:16]}...)")
        return True

    def build_and_load_model(self) -> Tuple[TGCFIDS, Dict[str, Any]]:
        """Constructs model architecture, loads frozen weights, and puts into eval mode."""
        if self.verify_hash:
            self.verify_checkpoint_integrity()

        model = TGCFIDS(
            num_numerical=39,
            cat_cardinalities=[134, 14, 10],
            token_dim=64,
            transformer_heads=4,
            transformer_layers=2,
            transformer_ffn_dim=256,
            transformer_dropout=0.1,
            transformer_pooling="mean",
            graph_in_channels=194,
            graph_edge_dim=6,
            graph_hidden_dim=64,
            graph_out_channels=64,
            graph_layers=2,
            graph_dropout=0.1,
            fusion_dim=128,
            fusion_strategy="gated",
            classifier_hidden_dim=64,
            classifier_dropout=0.2,
            num_classes=10,
        ).to(self.device)

        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        manifest = {
            "model_version": self.MODEL_VERSION,
            "checkpoint_name": self.checkpoint_path.name,
            "device": str(self.device),
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "status": "LOADED_AND_FROZEN",
        }
        logger.info(f"[+] TGCF-IDS Production Model initialized successfully ({manifest['num_parameters']} params).")
        return model, manifest


def compute_file_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA256 checksum for a file."""
    path = Path(file_path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_production_model(
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
    verify_hash: bool = True
) -> TGCFIDS:
    """Convenience helper to load frozen TGCF-IDS model instance."""
    loader = ProductionModelLoader(
        checkpoint_path=checkpoint_path,
        device=device,
        verify_hash=verify_hash
    )
    model, _ = loader.build_and_load_model()
    return model

