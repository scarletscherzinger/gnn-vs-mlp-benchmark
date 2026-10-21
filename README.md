# Evaluating the Utility of Graph Structure: Accuracy vs. Inference-Cost Benchmarks on Cora

## Executive Summary

In machine learning, leveraging graph topology (relational data between entities) often improves predictive performance over tabular or feature-only models. However, in production environments, graph neural networks (GNNs) introduce systems overhead such as neighbor data fetching and aggregation latency.This project rigorously benchmarks three architectures on the **Cora citation dataset** to quantify the trade-off between **predictive accuracy** and **systems performance (inference latency and model parameter scale):**

1. **Multi-Layer Perceptron (MLP):** Baseline non-graph model (feature-only).
2. **Graph Convolutional Network (GCN):** Full-neighborhood message-passing model.
3. **GraphSAGE (Sample and Aggregate):** Inductive representation learning via neighborhood aggregation.To ensure statistical validity, all models were evaluated across **10 distinct random seeds ($N=10$).**


## Key Experimental Results

| Model | Test Accuracy (%) | Parameters | Latency (ms) |
| :--- | :--- | :--- | :--- |
| **MLP (No Graph)** | 51.88 ± 3.15 | 23,063 | 4.05 ± 0.58 |
| **GCN** | 78.50 ± 1.42 | 23,063 | 6.15 ± 0.68 |
| **GraphSAGE** | 79.20 ± 1.25 | 46,103 | 56.28 ± 9.33 |


## Quantitative Takeaways & Systems Analysis 

1. **Does Graph Structure Help? (+28.39% Accuracy Boost)**

🔹**The Baseline Failure:** The MLP model ignores node connections (citation edges) and classifies papers based exclusively on bag-of-words text features. It yields a mediocre **51.88% test accuracy** with high variance (**$\pm 3.16$**).

🔹**The GNN Advantage:** Incorporating graph topology via GCN increases test accuracy to **80.27%** a net improvement of **+28.39** percentage points. Furthermore, standard deviation drops to **$\pm 0.65$**, demonstrating that relational structure acts as a powerful regularizer, dramatically improving model stability across different random initializations.

2. **Accuracy vs. Systems Inference Cost Trade-offs** 

🔹**GCN Efficiency:** GCN achieves peak performance while maintaining the exact same parameter count as the MLP (**23,063 parameters**). However, its inference latency increases from 3.03 ms to 9.65 ms (**$3.18\times$ slowdown**). This additional latency represents the computational cost of performing graph convolution matrix multiplications across the adjacency matrix.

🔹**GraphSAGE Scalability Analysis:** GraphSAGE achieves competitive accuracy (**79.44%**), but exhibits higher parameter scale (**46,103 parameters**) and increased inference overhead (**60.15 ms**) under transductive, single-batch processing. While GraphSAGE is designed for scalable mini-batching on massive dynamic graphs, full-graph inference on small datasets like Cora illustrates the trade-off of aggregation functions versus global message passing.Methodology & Experimental Rigor


## Methodology & Experimental Rigor

**Dataset Overview**

🔹**Dataset:** Cora (Planetoid benchmark dataset)

🔹**Nodes:** 2,708 academic papers (each represented by a 1,433-dimensional binary word feature vector)

🔹**Edges:** 10,556 citation links

🔹**Classes:** 7 machine learning sub-fields


## Experimental Design 

🔹 **Seed Protocol:** Multi-seed evaluation using 10 fixed random seeds (`[42, 100, 7, 99, 123, 456, 789, 321, 654, 987]`) to eliminate initialization bias and report standard deviation.

🔹 **Training Hyperparameters:** 200 epochs per run, Adam optimizer (`lr` = 0.01, `weight_decay` = 5e-4), Dropout rate $p = 0.5$.

🔹 **Systems Benchmark:** Inference latency measured as the warm-started mean over 100 evaluation forward passes. Parameter counts calculated via trainable model weight counts (`numel`).


## Repository Structure & Quickstart

```text
cora-graph-benchmark/
├── benchmark.py    # Complete PyTorch Geometric benchmark script
├── README.md       # Project documentation and systems analysis
└── results/        # Raw benchmark metrics and logs
```


## Reproducing Results

1. **Install Dependencies:** pip install torch torch-geometric numpy
2. **Execute Benchmark:** python benchmark.py


## Technical Stack
  
🔹**Framework:** PyTorch & PyTorch Geometric (torch_geometric)

🔹**Evaluation & Metrics:** NumPy, Python time profiling

🔹**Dataset API:** torch_geometric.datasets.Planetoid

