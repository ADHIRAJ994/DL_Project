# Adaptive Learned Image Compression with Graph Neural Networks (GLIC)

A research-oriented implementation and experimental study of **GLIC (Graph-based Learned Image Compression)**, an adaptive learned image compression framework that uses **Graph-based Feature Aggregation (GFA)** to model content-dependent relationships between image features.

This project combines the official GLIC implementation with reproducible inference experiments, GFA visualizations, controlled graph-connectivity ablations, and a simplified custom GFA prototype.

---

## Project Overview

Traditional learned image-compression networks commonly rely on fixed convolutional or window-based local connectivity. That structure is efficient, but it does not always adapt well to images containing very different spatial patterns such as edges, textures, repeated structures, and smooth regions.

GLIC addresses this by introducing **content-adaptive graph connectivity** inside the image-compression transform.

Instead of treating every feature location in exactly the same way, GLIC:

1. Represents feature-map locations as graph nodes.
2. Samples candidate neighbors at **local and global scales**.
3. Estimates local content complexity using **Sobel-based gradients**.
4. Allocates graph degree adaptively according to content.
5. Uses **cosine similarity** to identify relevant neighbors.
6. Builds an adaptive graph mask.
7. Performs **graph attention / feature aggregation** over the selected connections.

The result is a compression transform whose feature communication is more adaptive to image content.

---

## Objectives

The project has four main objectives:

- Understand the architecture and motivation of GLIC and GFA.
- Run the official pretrained GLIC models and measure rate–distortion performance.
- Study the effect of removing local or global graph connectivity through inference-time ablation.
- Implement and train a simplified custom `MyGFA` prototype to understand the core graph-aggregation idea independently of the full compression stack.

---

## High-Level Architecture

```text
                         GLIC Compression Pipeline

 Input Image x
      │
      ▼
 ┌───────────────┐
 │ Analysis      │
 │ Transform g_a │
 └───────┬───────┘
         │
         │ feature maps
         ▼
 ┌─────────────────────────────┐
 │ Graph-based Feature         │
 │ Aggregation (GFA)           │
 │                             │
 │ Local + Global Candidates   │
 │ → Content Complexity        │
 │ → Adaptive Degree           │
 │ → Cosine Similarity         │
 │ → Graph Attention           │
 └─────────────┬───────────────┘
               │
               ▼
        Latent representation y
               │
               ▼
 ┌─────────────────────────────┐
 │ Hyperprior / Entropy Model  │
 │ Probability estimation      │
 │ + context models            │
 └─────────────┬───────────────┘
               │
               ▼
       Entropy-coded bitstream
               │
               ▼
       Quantized latent y-hat
               │
               ▼
 ┌───────────────┐
 │ Synthesis     │
 │ Transform g_s │
 └───────┬───────┘
         │
         ▼
   Reconstructed image x-hat
```

The official model uses GFA blocks in multiple encoder and decoder stages while retaining the learned hyperprior and entropy-coding components used for practical learned compression.

---

## What is GFA?

**Graph-based Feature Aggregation (GFA)** is the central adaptive component.

### 1. Feature map → graph nodes

For a feature map of shape `B × C × H × W`, each spatial position can be treated as a graph node. The model therefore has `H × W` spatial nodes per sample.

### 2. Dual-scale candidate sampling

GFA builds two candidate sets:

- **Local candidates:** nearby spatial positions.
- **Global candidates:** sparse positions sampled across a wider spatial range.

This allows the model to combine fine local interactions with long-range relationships.

### 3. Content complexity

A Sobel operator estimates the strength of local image/feature gradients. High gradient energy corresponds to more complex regions such as edges and textures; low energy corresponds to smoother regions.

A representative complexity measure used by the implementation is an RMS aggregation of the Sobel gradient magnitude.

### 4. Adaptive degree

The estimated content complexity is used to determine how many graph connections a node should receive. The aim is to avoid giving every node an identical connectivity pattern.

### 5. Similarity-based neighbor selection

Candidate feature vectors are compared using cosine similarity. An adaptive graph mask selects the most relevant relationships while respecting the content-dependent connectivity budget.

### 6. Graph attention

For the selected edges, GFA computes attention/correlation scores using projected features, cosine similarity, a learnable scale, and relative-position information. Messages from relevant neighbors are then aggregated back into the node features.

---

## Why Local + Global Graphs?

The dual-scale design is important because images contain structure at multiple spatial scales.

**Local connectivity** is useful for:
- edges and boundaries,
- nearby textures,
- fine spatial detail.

**Global connectivity** is useful for:
- repeated patterns,
- distant but semantically related regions,
- long-range dependencies.

GLIC combines both rather than forcing every feature interaction to be purely local or purely global.

---

## Experimental Setup

### Test images

The project evaluates the official checkpoints on a 9-image test set:

```text
animals.jpg
building.jpg
city.jpg
flowers.jpg
food.jpg
indoor.jpg
landscape.jpg
people.jpg
road.jpg
```

### Official checkpoints evaluated

Three pretrained operating points were evaluated:

- `0.05checkpoint_best.pth.tar`
- `0.025checkpoint_best.pth.tar`
- `0.0067checkpoint_best.pth.tar`

The reported main results use the **actual `Compress` pipeline** (entropy-coded bitstream generation followed by decompression), rather than only the forward-pass rate estimate.

---

## Main Rate–Distortion Results

| Checkpoint (λ) | BPP | PSNR (dB) |
|---|---:|---:|
| **0.05** | **1.042813** | **37.0306** |
| **0.025** | **0.768519** | **35.2129** |
| **0.0067** | **0.420127** | **31.7742** |

### Interpretation

The three operating points show the expected **rate–distortion trade-off**:

- Increasing the compression rate improves reconstruction quality.
- Lower-bitrate operation reduces storage/transmission cost but decreases PSNR.
- Across the evaluated operating points, bitrate falls from **1.042813 BPP to 0.420127 BPP**, while PSNR falls from **37.0306 dB to 31.7742 dB**.

The corresponding R-D curve is included in `results/rd_curve_glic.png` when the project results are packaged.

---

## GFA Ablation Study

A controlled inference-time ablation was performed using the **same pretrained GLIC checkpoint** while changing the sampling mode of the GFA blocks:

| GFA Configuration | BPP | PSNR (dB) |
|---|---:|---:|
| **Official dual-scale** | **1.042561** | **37.0316** |
| **Local-only** | **1.034814** | **36.8585** |
| **Global-only** | **1.031651** | **36.7226** |

### Interpretation

The original dual-scale configuration gives the highest reconstruction quality in this experiment.

Relative to the official dual-scale setting:

- Local-only loses **0.1731 dB PSNR**.
- Global-only loses **0.3090 dB PSNR**.

The global-only configuration reaches a slightly lower bitrate, but it also incurs a larger quality reduction. The result therefore supports retaining both local and global graph connectivity rather than judging the configurations on bitrate alone.

> **Experimental caveat:** this is an **inference-time ablation on one pretrained checkpoint**. The ablated variants were not independently retrained, so the result should be interpreted as a controlled sensitivity study rather than a comparison of three independently optimized models.

---

## GFA Visualization

The project includes a visualization tool that exposes the graph behavior of the encoder stages. The generated maps include:

```text
stage1_complexity.png
stage1_local_degree.png
stage1_global_degree.png
stage2_complexity.png
stage2_local_degree.png
stage2_global_degree.png
```

These visualizations are intended to make the adaptive behavior of GFA interpretable:

- **Complexity maps** indicate spatial content variation.
- **Local degree maps** show local graph connectivity.
- **Global degree maps** show global graph connectivity.

The visualization demonstrates that graph structure can be inspected at the feature level rather than treating the compression transform as a black box.

---

## Custom MyGFA Prototype

To understand the GFA concept independently, a simplified experimental implementation named `MyGFA` was developed.

The prototype includes:

```text
Sobel complexity
      ↓
Local candidate sampling
      ↓
Global candidate sampling
      ↓
Cosine similarity
      ↓
Adaptive neighbor selection
      ↓
Global adaptive aggregation
      ↓
Residual + feed-forward refinement
```

A small CNN baseline was compared with the CNN + `MyGFA` prototype:

| Model | MSE | PSNR (dB) |
|---|---:|---:|
| **CNN Baseline** | **0.017222** | **17.6409** |
| **CNN + Custom MyGFA** | **0.018760** | **17.2753** |

### Important interpretation

This prototype is **not a replacement for the official GLIC model**. It is a simplified educational/research experiment with a tiny dataset and a much smaller architecture.

The experiment therefore demonstrates the process of implementing and testing an adaptive graph block, but it should not be used to claim that GFA is intrinsically worse than CNNs. The official GLIC experiments above are the primary compression results.

---

## Repository Structure

```text
GLIC1/
│
├── glic_model/
│   ├── layers/
│   ├── models/
│   └── utils/
│
├── test_images/
│
├── results/
│   ├── gfa_visualization/
│   ├── rd_005.json
│   ├── rd_0025.json
│   ├── rd_0067.json
│   └── rd_curve_glic.png
│
├── test.py
├── train.py
├── Meter.py
│
├── ablation.py
├── visualize_gfa.py
├── my_gfa.py
├── custom_compression.py
├── train_prototype.py
│
├── README.md
└── requirements.txt
```

Large pretrained checkpoints may be kept outside Git and alongside the project ZIP when a fully runnable offline copy is required.

Expected checkpoint files:

```text
0.05checkpoint_best.pth.tar
0.025checkpoint_best.pth.tar
0.0067checkpoint_best.pth.tar
```

---

## Installation

A Python environment with PyTorch and the dependencies listed in `requirements.txt` is required.

Example:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For a CUDA-enabled PyTorch installation, use a PyTorch build compatible with the GPU/driver available on the target machine.

> The pretrained GLIC checkpoints are large binary files and are intentionally excluded from normal Git version control.

---

## Running the Official Evaluation

Evaluate one checkpoint:

```bash
python test.py --checkpoint .\0.05checkpoint_best.pth.tar --image-dir .\test_images --num-workers 0 --save-json .\results\rd_005.json
```

```bash
python test.py --checkpoint .\0.025checkpoint_best.pth.tar --image-dir .\test_images --num-workers 0 --save-json .\results\rd_0025.json
```

```bash
python test.py --checkpoint .\0.0067checkpoint_best.pth.tar --image-dir .\test_images --num-workers 0 --save-json .\results\rd_0067.json
```

For a combined multi-checkpoint evaluation:

```bash
python test.py --checkpoint .\0.05checkpoint_best.pth.tar .\0.025checkpoint_best.pth.tar .\0.0067checkpoint_best.pth.tar --image-dir .\test_images --num-workers 0 --save-json .\results\rd_results.json
```

---

## Running the GFA Analysis

Run the inference-time sampling ablation:

```bash
python ablation.py
```

Generate GFA complexity and degree visualizations:

```bash
python visualize_gfa.py
```

---

## Running the Custom Prototype

The simplified implementation is contained in:

```text
my_gfa.py
custom_compression.py
train_prototype.py
```

Train the small prototype with:

```bash
python train_prototype.py
```

The prototype is intended for understanding and experimentation, not for reproducing the full computational scale of the official GLIC model.

---

## Reproducibility Notes

To keep the experimental claims precise:

- **Official GLIC results** come from pretrained GLIC checkpoints evaluated on the project's 9-image test set.
- **R-D results** use actual entropy-coded `Compress` measurements.
- **GFA ablation** is inference-time and uses the same pretrained checkpoint for all sampling variants.
- **Custom MyGFA** is a simplified prototype trained on a very small dataset and should be treated as an exploratory experiment.

These distinctions are important when interpreting the results.

---

## Key Takeaways

1. **GLIC makes feature communication content-adaptive by constructing graphs over spatial feature locations.**
2. **GFA combines local and global candidate interactions instead of relying on a single fixed connectivity pattern.**
3. **Sobel-based complexity and similarity-aware graph selection provide an interpretable mechanism for adaptive connectivity.**
4. **The pretrained model demonstrates a clear rate–distortion trade-off across multiple operating points.**
5. **Removing either local or global graph connectivity reduces PSNR in the controlled inference-time ablation.**
6. **The custom MyGFA prototype provides an independent implementation-level study of the main graph aggregation idea.**

---

## Reference

This project is based on the paper:

**Adaptive Learned Image Compression with Graph Neural Networks**, CVPR 2026.

Paper:  
https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Adaptive_Learned_Image_Compression_with_Graph_Neural_Networks_CVPR_2026_paper.html

Official implementation:  
https://github.com/UnoC-727/GLIC

Please preserve the original project's attribution and license information when redistributing or modifying the implementation.

---

## Project Contribution

This repository is presented as an experimental project built around the official GLIC implementation. The work performed here includes:

- setting up and running the official GLIC model,
- evaluating three pretrained rate–distortion operating points,
- generating GFA interpretability visualizations,
- performing local-only and global-only graph ablations,
- implementing a simplified custom `MyGFA` block,
- training a small prototype for comparison,
- organizing quantitative results and visual evidence for project presentation.

---

## Final Conclusion

GLIC demonstrates how **graph-based, content-adaptive feature aggregation** can be integrated into learned image compression. Rather than imposing identical spatial interactions everywhere, GFA adapts graph connectivity to feature content and combines local and long-range relationships.

The experiments in this project reproduce the official pretrained compression pipeline, characterize its rate–distortion behavior, and provide an ablation-based study of its dual-scale graph design. The additional custom prototype reinforces the main architectural idea by implementing an independently understandable GFA-style block.

**Result:** the project provides an end-to-end experimental study of graph-based learned image compression, from architecture and graph construction to compression metrics, ablation, visualization, and reproducible evaluation.
