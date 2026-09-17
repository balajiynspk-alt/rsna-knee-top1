# TGCF-IDS: Temporal Graph Contrastive Feature-Transformer Intrusion Detection System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-2.4%2B-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An academic research framework designed for state-of-the-art Network Intrusion Detection Systems (NIDS). **TGCF-IDS** couples **Temporal Graph Neural Networks** with **Feature Transformers** via **Self-Supervised / Supervised Contrastive Representation Learning** to detect sophisticated, multi-stage cyber attacks across dynamic network topologies.

---

## 📁 Repository Directory Structure

The repository is organized following reproducible, publication-ready research best practices:

```text
TGCF-IDS/
│
├── data/                       # Dataset storage across lifecycle stages
│   ├── raw/                    # Untouched, original dataset files (e.g., CSV, PCAP)
│   ├── processed/              # Cleaned, encoded, scaled tabular tensors
│   └── graphs/                 # Serialized PyTorch Geometric temporal graph snapshots (.pt)
│
├── configs/                    # Experiment-specific YAML configurations (ablations, baselines)
│   └── default_config.yaml     # Baseline configuration reference
│
├── notebooks/                  # Jupyter notebooks for exploratory data analysis (EDA) & visualization
│
├── src/                        # Core source code modules for the research pipeline
│   ├── data/                   # Dataset loaders, batch iterators, and streaming samplers
│   ├── preprocessing/          # Tabular cleaning, missing value imputation, robust scaling
│   ├── graphs/                 # Flow-to-graph transformation & temporal snapshot windowing
│   ├── models/                 # Feature-Transformer, Graph Encoder (GAT/TGAT), & Contrastive Heads
│   ├── training/               # Multi-task training loops, contrastive loss (InfoNCE/SupCon), optimizers
│   ├── evaluation/             # Metrics calculation (AUROC, AUPRC, F1-macro, FAR), PR/ROC curves
│   └── utils/                  # Cross-platform pathlib managers, seeders, and environment diagnostics
│
├── tests/                      # Automated unit and integration test suite (pytest)
│
├── scripts/                    # Standalone utility & execution scripts (diagnostics, export tools)
│   └── check_env.py            # Hardware & deep learning dependency diagnostic tool
│
├── experiments/                # Artifacts, logs, and hyperparameter logs for paper experiments
│
├── results/                    # Publication assets and generated artifacts
│   ├── figures/                # Publication-ready plots, ROC/PR curves, t-SNE embeddings (.pdf, .png)
│   ├── tables/                 # LaTeX and Markdown formatted evaluation metric tables
│   ├── checkpoints/            # Saved PyTorch model weights (.pt, .pth)
│   ├── logs/                   # Training logs, TensorBoard / CSV event runs
│   └── final/                  # Final aggregated results for the research paper
│
├── requirements.txt            # Python dependencies specification
├── README.md                   # Project documentation, setup guide, and directory explanation
├── .gitignore                  # Git tracking exclusions for large datasets & checkpoints
├── main.py                     # Main CLI execution entry point
├── config.yaml                 # Master configuration file (hyperparameters, paths, flags)
└── pytest.ini                  # Pytest configuration file
```

---

## 🔬 Directory Explanations & Responsibilities

| Directory | Purpose & Research Responsibility |
| :--- | :--- |
| **`data/raw/`** | Stores raw benchmark datasets (e.g., UNSW-NB15, CIC-IDS2017, TON_IoT). Kept strictly read-only to guarantee reproducibility. |
| **`data/processed/`** | Contains standardized tabular splits (train/val/test) after one-hot/target encoding and robust scaling. |
| **`data/graphs/`** | Stores constructed temporal graph objects (nodes = IP/Hosts, edges = network flows with multi-dimensional temporal attributes). |
| **`configs/`** | Holds modular configuration files for ablation studies, varying hyperparameter sets, and different model architectures. |
| **`notebooks/`** | Dedicated to interactive research exploration, feature correlation heatmaps, graph degree distributions, and visual prototypes. |
| **`src/data/`** | Implements custom PyTorch & PyG Dataset / DataLoader classes capable of handling tabular and graph streams. |
| **`src/preprocessing/`** | Implements leakage-free transformations (fit on train split only, transform on val/test). |
| **`src/graphs/`** | Translates network flow records into sliding-window temporal graph snapshots suitable for dynamic graph neural networks. |
| **`src/models/`** | Houses model architectures: Feature-Transformer for tabular flow representation, Graph Neural Encoders for topology, and Contrastive Projection heads. |
| **`src/training/`** | Encapsulates training loops, contrastive loss implementations (InfoNCE, Supervised Contrastive Loss), learning rate schedulers, and early stopping. |
| **`src/evaluation/`** | Computes rigorous security metrics: Detection Rate (Recall), Precision, F1-Macro, False Alarm Rate (FAR), Area Under ROC (AUROC), and Precision-Recall Curve (AUPRC). |
| **`src/utils/`** | Shared utilities including deterministic seed management and cross-platform `pathlib` filesystem resolution. |
| **`tests/`** | Pytest-based unit testing for data loaders, graph transformations, loss functions, and tensor shapes. |
| **`scripts/`** | Standalone command-line utilities such as environment diagnostics, data downloading scripts, and batch evaluation runners. |
| **`experiments/`** | Tracks metadata, hyperparameter search grids, and execution manifests for each experiment run. |
| **`results/`** | Hierarchical directory storing paper-ready figures (`figures/`), LaTeX summary tables (`tables/`), model checkpoints (`checkpoints/`), and execution logs (`logs/`). |

---

## ⚙️ Environment Setup & Installation Guide (Windows 11 PowerShell)

Follow these exact steps in **PowerShell** to set up a dedicated virtual environment with GPU acceleration.

### Step 1: Navigate to Project Root
```powershell
cd C:\unsw\TGCF-IDS
```

### Step 2: Create a Virtual Environment
Using standard Python `venv`:
```powershell
python -m venv venv
```

Activate the virtual environment:
```powershell
.\venv\Scripts\Activate.ps1
```

*(Note: If PowerShell execution policy restricts script execution, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

---

### Step 3: Install PyTorch with CUDA Acceleration

For **NVIDIA GPU (CUDA 12.1)** support:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

*(For CPU-only fallback if no NVIDIA GPU is present):*
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

---

### Step 4: Install PyTorch Geometric (PyG) and Project Dependencies

Install **PyTorch Geometric**:
```powershell
pip install torch-geometric
```

Install remaining research dependencies:
```powershell
pip install -r requirements.txt
```

---

### Step 5: Run the Hardware & Environment Diagnostic

Execute the diagnostic script to verify Python, PyTorch, CUDA, and PyG:
```powershell
python scripts/check_env.py
```

*Or via the main CLI entry point:*
```powershell
python main.py --check-env
```

---

### Step 6: Verify with Automated Tests

Run the initial smoke test suite:
```powershell
pytest
```

---

## 🛠️ Cross-Platform Path Handling

All paths across the codebase strictly use Python's `pathlib.Path` via `src.utils.ProjectPaths`. No hardcoded backslashes (`\`) or platform-specific separators are used, guaranteeing identical behavior across Windows 11, Linux clusters (SLURM / HPC), and macOS.
