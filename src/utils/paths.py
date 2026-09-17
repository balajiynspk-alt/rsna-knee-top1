"""
Centralized cross-platform path management using pathlib.
Ensures seamless portability across Windows, Linux, and macOS.
"""

from pathlib import Path


class ProjectPaths:
    """Project-wide cross-platform path resolver."""

    # Project root is 2 levels up from src/utils/paths.py -> src -> TGCF-IDS
    ROOT_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # Data paths
    DATA_DIR: Path = ROOT_DIR / "data"
    DATA_RAW: Path = DATA_DIR / "raw"
    DATA_PROCESSED: Path = DATA_DIR / "processed"
    DATA_GRAPHS: Path = DATA_DIR / "graphs"

    # Configs
    CONFIGS_DIR: Path = ROOT_DIR / "configs"
    DEFAULT_CONFIG: Path = ROOT_DIR / "config.yaml"

    # Notebooks & Scripts
    NOTEBOOKS_DIR: Path = ROOT_DIR / "notebooks"
    SCRIPTS_DIR: Path = ROOT_DIR / "scripts"
    SRC_DIR: Path = ROOT_DIR / "src"
    TESTS_DIR: Path = ROOT_DIR / "tests"

    # Experiments and Results
    EXPERIMENTS_DIR: Path = ROOT_DIR / "experiments"
    RESULTS_DIR: Path = ROOT_DIR / "results"
    RESULTS_FIGURES: Path = RESULTS_DIR / "figures"
    RESULTS_TABLES: Path = RESULTS_DIR / "tables"
    RESULTS_CHECKPOINTS: Path = RESULTS_DIR / "checkpoints"
    RESULTS_LOGS: Path = RESULTS_DIR / "logs"
    RESULTS_FINAL: Path = RESULTS_DIR / "final"

    @classmethod
    def ensure_directories(cls) -> None:
        """Create all required project directories if they do not exist."""
        directories = [
            cls.DATA_RAW,
            cls.DATA_PROCESSED,
            cls.DATA_GRAPHS,
            cls.CONFIGS_DIR,
            cls.NOTEBOOKS_DIR,
            cls.SCRIPTS_DIR,
            cls.EXPERIMENTS_DIR,
            cls.RESULTS_FIGURES,
            cls.RESULTS_TABLES,
            cls.RESULTS_CHECKPOINTS,
            cls.RESULTS_LOGS,
            cls.RESULTS_FINAL,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
