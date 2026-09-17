"""
Basic smoke tests to verify project paths, cross-platform integrity, and importability.
"""

import sys
import pytest
from pathlib import Path


def test_python_version():
    """Verify that Python version is at least 3.9."""
    assert sys.version_info >= (3, 9), f"Python 3.9+ required, found {sys.version}"


def test_project_paths_resolution():
    """Verify that ProjectPaths resolves valid directories across platforms."""
    from src.utils.paths import ProjectPaths

    assert ProjectPaths.ROOT_DIR.exists(), "Root directory should exist"
    assert ProjectPaths.ROOT_DIR.is_dir(), "Root directory should be a valid directory"
    assert ProjectPaths.SRC_DIR.exists(), "src directory should exist"
    assert ProjectPaths.CONFIGS_DIR.exists(), "configs directory should exist"
    assert ProjectPaths.TESTS_DIR.exists(), "tests directory should exist"


def test_config_file_exists():
    """Verify that config.yaml exists at project root."""
    from src.utils.paths import ProjectPaths

    assert ProjectPaths.DEFAULT_CONFIG.exists(), "config.yaml should exist at project root"


def test_src_submodules_importable():
    """Verify all submodules in src package are discoverable and importable."""
    import src
    import src.data
    import src.preprocessing
    import src.graphs
    import src.models
    import src.training
    import src.evaluation
    import src.utils

    assert src.__version__ is not None
