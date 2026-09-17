# TGCF-IDS: Research Model Contract & Production Interface Specification

**Contract Version**: `TGCF-IDS-production-compatible-v1`  
**Dataset Reference**: UNSW-NB15 Benchmark (ACCS)  
**Baseline Model**: Dual-Branch Temporal Graph Contrastive Feature-Transformer  
**Isolation Status**: **FROZEN RESEARCH BASELINE (READ-ONLY)**  
**Audit Date**: September 17, 2026  

---

## 1. Research Repository & Model Inventory

The production system interfaces with the frozen research artifacts located at:

| Artifact | Research Path | Specifications / Hash |
| :--- | :--- | :--- |
| **Primary Checkpoint** | `results/checkpoints/final_tgcf_ids_model.pt` | SHA256: `f6a85511ef0d44bed49a481c4a720a9173c57ec9f91ed64192576b019cf8cd54` (Size: 1,023,514 B) |
| **Best Checkpoint** | `results/checkpoints/best_tgcf_ids.pt` | SHA256: `d08e53ba75d8f82af21d0d348e8835b6386c2ca52bb2c758e6fb71532f038c9c` (Size: 3,183,946 B) |
| **Graph Pretrained** | `results/checkpoints/graph_pretrained.pt` | SHA256: `cbb730a91f42d250d4f3b1451f215d5ec37e381640a20e2ef5b1e6e96fa6bcda` (Size: 233,866 B) |
| **Fitted Preprocessor** | `data/processed/preprocessor.joblib` | Fitted `LeakageSafePreprocessor` object (Joblib v1.0.0) |
| **Metadata Manifest** | `data/processed/metadata.json` | Exact feature names, scaling medians/IQRs, vocabularies |
| **Research Architecture**| `src/models/tgcf_ids.py` | PyTorch `TGCFIDS` module definition |

---

## 2. Multi-Class Label Taxonomy

The classification head outputs logits across **$C = 10$ mutually exclusive traffic categories**:

```python
CLASS_MAPPING = {
    0: "Normal",          # Benign traffic
    1: "Analysis",        # Web / vulnerability scan probe
    2: "Backdoor",        # Persistent backdoor access
    3: "DoS",             # Denial of Service flood
    4: "Exploits",        # Known vulnerability exploitation
    5: "Fuzzers",         # Protocol fuzzing & malformed packets
    6: "Generic",         # High-rate cipher / payload floods
    7: "Reconnaissance",  # Port / IP scanning & sweeping
    8: "Shellcode",       # Small payload exploit / execution
    9: "Worms"            # Self-propagating worm infection
}
```

---

## 3. Input Feature Schema & Exact Ordering

The research model accepts a total of **42 tabular input features** ($39$ continuous/discrete numerical features + $3$ categorical fields).

### 3.1 Numerical Features (39 Continuous Features in Exact Order)
1. `dur` (Duration in seconds)
2. `spkts` (Source-to-destination packet count)
3. `dpkts` (Destination-to-source packet count)
4. `sbytes` (Source-to-destination transaction bytes)
5. `dbytes` (Destination-to-source transaction bytes)
6. `rate` (Packets transmitted per second)
7. `sttl` (Source time to live)
8. `dttl` (Destination time to live)
9. `sload` (Source bits per second)
10. `dload` (Destination bits per second)
11. `sloss` (Source packet retransmissions/losses)
12. `dloss` (Destination packet retransmissions/losses)
13. `sinpkt` (Source inter-packet arrival time in ms)
14. `dinpkt` (Destination inter-packet arrival time in ms)
15. `sjit` (Source jitter in ms)
16. `djit` (Destination jitter in ms)
17. `swin` (Source TCP window advertisement)
18. `stcpb` (Source TCP sequence number)
19. `dtcpb` (Destination TCP sequence number)
20. `dwin` (Destination TCP window advertisement)
21. `tcprtt` (TCP round-trip time: `synack` + `ackdat`)
22. `synack` (TCP SYN-to-SYNACK duration)
23. `ackdat` (TCP SYNACK-to-ACK duration)
24. `smean` (Mean flow packet size from source)
25. `dmean` (Mean flow packet size from destination)
26. `trans_depth` (HTTP request/response pipelining depth)
27. `response_body_len` (HTTP body content length)
28. `ct_srv_src` (Count of connections to same service from source)
29. `ct_state_ttl` (Count of connection state & TTL combinations)
30. `ct_dst_ltm` (Count of connections to destination in last 100)
31. `ct_src_dport_ltm` (Count of connections from source to dest port)
32. `ct_dst_sport_ltm` (Count of connections to destination from sport)
33. `ct_dst_src_ltm` (Count of connections between source and destination)
34. `is_ftp_login` (Binary flag: 1 if FTP session authenticated)
35. `ct_ftp_cmd` (Count of FTP commands executed)
36. `ct_flw_http_mthd` (Count of HTTP methods in connection)
37. `ct_src_ltm` (Count of connections from source in last 100)
38. `ct_srv_dst` (Count of connections to service at destination)
39. `is_sm_ips_ports` (Binary flag: 1 if src and dst IP/ports identical)

### 3.2 Categorical Features (3 Categorical Fields in Exact Order)
1. `proto` (Transport/Network Protocol: 133 unique training categories + 1 `<UNK>` index = cardinality 134)
2. `service` (Application Layer Service: 13 unique training categories + 1 `<UNK>` index = cardinality 14)
3. `state` (Connection State: 9 unique training categories + 1 `<UNK>` index = cardinality 10)

### 3.3 Prohibited Feature Fields (Must NEVER enter feature vector)
- `label` (Binary ground truth target)
- `attack_cat` (Multi-class ground truth target)
- `id` (Sequential row identifier)
- `y_multiclass` / `y_binary`
- `srcip`, `dstip`, `sport`, `dsport` (Raw IP addresses / ports must remain in flow metadata and graph topology, never in tabular feature tensors)

---

## 4. Preprocessing & Scaling Rules

1. **Numerical Normalization**: Features are scaled using `RobustScaler` fitted on training split medians and IQRs:
   $$x_{scaled} = \frac{x - Q_2(X_{train})}{Q_3(X_{train}) - Q_1(X_{train})}$$
2. **Missing Numerical Handling**: Imputed with median of training distribution.
3. **Categorical Encoding**: Mapped to integer indices ($0 \dots |\mathcal{V}_k|-1$). Any unknown category not seen in training is assigned index $0$ (`<UNK>`).
4. **Dense Representation Vector (`x_dense`)**: Concatenation of $39$ scaled numericals + $155$ one-hot protocol/service/state categories = $\mathbb{R}^{194}$.

---

## 5. Temporal Graph Structure & Snapshot Definition

1. **Snapshot Window**: $\Delta t = 60.0\text{ seconds}$.
2. **Nodes**: Communicating IP host entities active within window $[t, t + \Delta t)$.
3. **Directed Edges**: $e_{uv} = (u, v)$ instantiated if host $u$ sends traffic to host $v$ within snapshot $t$.
4. **Edge Attributes ($d_e = 6$)**: Continuous 6-dimensional flow summary:
   $$\mathbf{e}_{uv} = \left[ \text{dur}, \text{sbytes}, \text{dbytes}, \text{spkts}, \text{dpkts}, \text{rate} \right]^\top$$
   Normalized via log-transform and standard scaling: $\mathbf{e}_{uv} \leftarrow \text{StandardScaler}(\log(1 + \mathbf{e}_{uv}))$.

---

## 6. Model Architecture & Expected Tensor Dimensions

```
+---------------------------------------------------------------------------------------------------+
| MODEL MODULE              | INPUT TENSOR DIMENSIONS                   | OUTPUT TENSOR DIMENSIONS  |
+---------------------------+-------------------------------------------+---------------------------+
| FeatureTokenizer          | x_num: [N, 39], x_cat: [N, 3]             | tokens: [N, 42, 64]       |
| FeatureTransformer        | tokens: [N, 42, 64]                       | h_feat: [N, 64]           |
| TemporalGraphSAGE         | x_dense: [N_node, 194], edge_index: [2, E]| h_graph: [N_node, 64]     |
|                           | edge_attr: [E, 6]                         |                           |
| CrossModalFusion (Gated)  | h_feat: [N, 64], h_graph: [N, 64]         | h_fused: [N, 128]         |
|                           |                                           | gate: [N, 128]            |
| Classification Head       | h_fused: [N, 128]                         | logits: [N, 10]           |
+---------------------------------------------------------------------------------------------------+
```

### Parameter Specification:
- `num_numerical`: 39
- `cat_cardinalities`: `[134, 14, 10]`
- `token_dim`: 64
- `transformer_heads`: 4
- `transformer_layers`: 2
- `transformer_ffn_dim`: 256
- `graph_in_channels`: 194
- `graph_edge_dim`: 6
- `graph_hidden_dim`: 64
- `graph_out_channels`: 64
- `graph_layers`: 2
- `fusion_dim`: 128
- `fusion_strategy`: `"gated"`
- `classifier_hidden_dim`: 64
- `num_classes`: 10
- Total Parameters: **248,714**

---

## 7. Production Parity Acceptance Gate

The production model adapter is validated if and only if for identical prepared flow tensors $\mathbf{X}$:
$$\max | \text{logits}_{prod}(\mathbf{X}) - \text{logits}_{research}(\mathbf{X}) | < 10^{-5}$$
$$\text{argmax}(\text{logits}_{prod}(\mathbf{X})) == \text{argmax}(\text{logits}_{research}(\mathbf{X})) \quad \forall \text{ test flows}$$
