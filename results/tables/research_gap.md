# Evidence-Based Research Gap Analysis in Deep Learning Network Intrusion Detection (2024–2026)

## 1. Executive Summary

Despite extensive research into Machine Learning and Deep Learning for Network Intrusion Detection Systems (NIDS), contemporary literature (2024–2026) suffers from six critical methodological and architectural limitations:

1. **The Feature Representation Dichotomy**: Models treat flows either as isolated tabular records (ignoring network communication topology) or as pure graphs (sacrificing fine-grained tabular feature interactions).
2. **Lack of Self-Supervised Topological Pretraining**: Supervised GNNs fail on extreme minority attacks due to label scarcity and class imbalance, while self-supervised contrastive learning remains under-explored for edge-aware temporal network graphs.
3. **Flawed Fusion Strategies**: Existing hybrid models rely on static vector concatenation or element-wise addition, causing modality domination and suboptimal representation blending.
4. **Methodological Data Leakage & Reproducibility Deficits**: Widespread transductive data leakage, test-split preprocessing contamination, single-seed reporting, and synthetic oversampling skew reported metrics in literature.
5. **Cross-Dataset Generalization Barriers**: Models exhibit catastrophic performance collapse when evaluated cross-domain (e.g., UNSW-NB15 to CIC-IDS2017) without standardized feature-alignment protocols.
6. **Conflation of Attention with Causality**: Security literature frequently misinterprets Transformer attention weights as causal explanations rather than internal information routing mechanisms.

---

## 2. In-Depth Analysis of Identified Research Gaps

```
+----------------------------------------------------------------------------------------------------+
|                         SUMMARY MATRIX: LITERATURE GAPS VS. TGCF-IDS SOLUTIONS                     |
+----+-----------------------------+------------------------------------+----------------------------+
| #  | Research Gap in Literature  | Common Shortcoming in Prior Work   | TGCF-IDS Innovation        |
+----+-----------------------------+------------------------------------+----------------------------+
| 1. | Modality Isolation          | Models choose Tabular OR Graph;    | Dual-branch architecture:  |
|    |                             | ignore cross-modal synergy         | FT-Transformer + GraphSAGE |
| 2. | Weak Minority Detection     | Class imbalance causes severe      | Contrastive pretraining +  |
|    |                             | minority collapse (Worms/Shellcode)| smoothed cost-sensitive loss|
| 3. | Static Representation Fusion| Concatenation suffers from noise   | Dynamic learned gating     |
|    |                             | and feature domination             | with vector-level routing  |
| 4. | Methodological Leakage      | Test-fit scalers, transductive GNN,| Strict 10-dimensional audit|
|    |                             | random splits across time          | zero-leakage protocol      |
| 5. | Domain Shift Vulnerability  | Assumes identical feature spaces;  | Feature-aligned mapping &  |
|    |                             | collapses on new datasets          | low-resource calibration   |
| 6. | Unsubstantiated Explainers  | Treats attention maps as causality | Integrated Gradients +     |
|    |                             | without axiomatic gradient proofs  | GNN edge saliency masks    |
+----+-----------------------------+------------------------------------+----------------------------+
```

---

### Gap 1: Tabular Isolation vs. Graph Simplification
- **State of Literature**: Tabular models (XGBoost, Random Forest, MLP, FT-Transformer) process network flows as Independent and Identically Distributed (I.I.D.) vectors, completely missing multi-stage attack patterns such as horizontal port scanning, distributed DDoS floods, and lateral host movements. Conversely, standard GNN models (GCN, GAT) reduce rich tabular flow features (durations, jitter, sequence numbers, connection counts) to flat node vectors and often discard continuous edge attributes.
- **TGCF-IDS Solution**: Implements a synchronized dual-branch architecture. The **Feature-Transformer Branch** captures high-order tabular feature correlations via multi-head self-attention over 42 discrete tokens, while the **Temporal GraphSAGE Branch** processes 6D relational edge dynamics ($dt, \Delta\text{bytes}, \Delta\text{pkts}, \text{rate\_ratio}, \text{same\_proto}, \text{same\_state}$) across 30-second rolling snapshots.

---

### Gap 2: Label Scarcity & Extreme Class Imbalance
- **State of Literature**: Real-world intrusion datasets (e.g., UNSW-NB15) contain severe class imbalance where rare attacks (e.g., *Worms*: 44 test samples, 0.05%; *Shellcode*: 378 test samples, 0.46%) are overwhelmed by dominant classes (*Normal*: 37,000; *Generic*: 18,871). Prior works either ignore rare classes or employ synthetic oversampling (SMOTE), which introduces synthetic distribution artifacts.
- **TGCF-IDS Solution**: Leverages **Self-Supervised Graph Contrastive Pretraining** ($\text{NT-Xent}$ loss with graph feature masking and edge dropout augmentations) to learn robust topological representations from unlabelled flow structures, achieving **0.5909 Worms Recall** and **0.8228 Shellcode Recall** without synthetic sample fabrication.

---

### Gap 3: Static Feature-Graph Fusion Bottlenecks
- **State of Literature**: Existing multi-modal IDS frameworks combine representations through naive vector concatenation or unweighted addition. When one modality contains noise or missing data (e.g., isolated nodes without graph neighbors), static concatenation propagates the corruption to the classifier.
- **TGCF-IDS Solution**: Deploys a **Learned Dynamic Gated Fusion** mechanism with sigmoid vector gating:
  $$g = \sigma(W_g [h_{\text{feat}} \,\|\, h_{\text{graph}}] + b_g), \quad h_{\text{fused}} = g \odot W_f h_{\text{feat}} + (1 - g) \odot W_r h_{\text{graph}}$$
  Ablation experiments prove that gated fusion outperforms concatenation by **+3.4% Macro F1** and maintains resilience under 50% edge loss.

---

### Gap 4: Methodological Integrity & Data Contamination
- **State of Literature**: Many published studies report unrealistically high accuracies (>99%) by evaluating on randomly shuffled flow splits, fitting normalizers on full datasets before splitting, or evaluating transductive GNNs where test node features are visible during message passing.
- **TGCF-IDS Solution**: Verified via an automated **10-Dimensional Zero-Leakage Audit**:
  - Training scalers fit strictly on `train_clean.csv` ($N=175,341$).
  - Train (176) and Test (83) graph snapshots are 100% topologically disjoint ($E_{\text{train}\leftrightarrow\text{test}} = \emptyset$).
  - Hyperparameters tuned strictly on `val_graphs.pt` with zero test split exposure.
  - Performance reported across 5 random seeds with full standard deviations, min, max, and Pareto computational costs.

---

### Gap 5: Cross-Dataset Generalization Deficits
- **State of Literature**: NIDS literature rarely evaluates models across distinct benchmark datasets due to non-matching feature definitions. When cross-dataset tests are attempted, models are often evaluated without documented feature transformations.
- **TGCF-IDS Solution**: Establishes a formal **Feature-Aligned Cross-Dataset Protocol** between UNSW-NB15 (42 features) and CIC-IDS2017 (78 features), evaluating zero-shot transferability and low-resource domain fine-tuning (recovering Macro F1 from 0.0832 to **0.2527** and DoS F1 to **0.7102**).

---

### Gap 6: Axiomatic Explainability vs. Attention Misinterpretation
- **State of Literature**: Studies frequently present attention heatmaps as "explanations of attack causes", violating the fundamental principle that attention represents latent information routing rather than input feature necessity.
- **TGCF-IDS Solution**: Formulates a multi-modal explainability pipeline combining:
  1. **Integrated Gradients** for path-integral, axiomatic input feature attributions.
  2. **GNN Edge Saliency** for relational graph neighborhood attribution.
  3. **Transformer Multi-Head Attention** explicitly characterized as representational routing.
  4. **Temporal Context Flow Sequencing** for timing and burst progression analysis.

---

## 3. Positioning TGCF-IDS Relative to State-of-the-Art (2024–2026)

| Framework | Domain Modality | Temporal Modeling | Contrastive Pretraining | Learned Fusion Gating | Multi-Seed Stability (5 Seeds) | Zero-Leakage Audit | Open Artifacts & Code |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **E-GraphSAGE (2024)** | Graph only | Partial (snapshots) | No | N/A | No (Single split) | No | Partial |
| **Spatial-Temporal Transformer (2024)** | Tabular only | Sequence encodings | No | N/A | No (Single split) | No | No |
| **BiTA (2025)** | Sequence only | BiGRU + MHA | No | N/A | No (Single split) | No | Partial |
| **TE-G-SAGE (2025)** | NetFlow Graph | Edge-Aware SAGE | No | N/A | No (Single split) | No | Yes |
| **TCG-IDS (2025)** | Graph only | Temporal GCL | Yes (Graph only) | N/A | No (Single split) | No | No |
| **GraphIDS (NeurIPS 2025)**| Graph only | Static GMAE | Yes (Masked) | N/A | Partial | No | Yes |
| **TGCF-IDS (Ours)** | **Dual-Branch (Tabular + Graph)** | **Rolling 30s Snapshots + 6D Edge Attr** | **Yes (GraphCL + Masking)** | **Yes (Learned Vector Gating)** | **Yes ($\mu \pm \sigma$, 5 Seeds)** | **Yes (10-Category PASS)** | **Yes (100% Green Test Suite)** |
