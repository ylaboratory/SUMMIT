<div style="display: flex; align-items: center;">
    <img src="./imgs/MINMF.svg" alt="SUMMIT-NMF" width="55" height="55" style="margin-right: 10px;">
    <h1 style="margin: 0; font-size: 2rem; line-height: 55px;">SUMMIT-NMF</h1>
</div>
**SUMMIT-NMF: Semi-supervised Unified Multi-target Integration Technique using NMF**


**Table of Conents:**

- [1. Installation](#1-installation)
- [2. Usage](#2-usage)
  - [Model types](#model-types)
  - [Optimization](#optimization)
  - [Training a model](#training-a-model)
- [3. More examples and analysis](#3-more-examples-and-analysis)
- [4. Citation](#4-citation)

## 1. Installation

## 2. Usage

### Model types

SUMMIT-NMF includes two primary models: `UnionSummitNMF` and `SeparatedSummitNMF`.  

- `UnionSummitNMF` represents each sample embedding as a linear combination of the sum of all associated label embeddings.  
- `SeparatedSummitNMF` enforces each sample embedding to be closer to its own label embeddings.  

Each approach has its own advantages. For more details, please refer to the paper.  

The table below lists the implemented classes associated with these two models. Classes prefixed with **Fast** preload the entire dataset onto the device, which is beneficial for small datasets and most use cases. However, for large datasets that cannot fit into device memory, users can opt for the standard classes, which support mini-batch processing and data swapping between the CPU and the device. While this allows handling larger datasets, it significantly increases computation time due to repeated data retrieval and updates.

| Class name | Model type | Class difference description |
| --- | --- | --- |
| `UnifiedSummitNMF` | `"unified"` or `"u"` | Unified version. Supports large datasets and mini-batch processing. |
| `SeperatedSummitNMF` | `"seperated"` or `"s"` | Seperated version. Supports large datasets and mini-batch processing. |
| `FastUnifiedSummitNMF` | `"fast_unified"` or `"fu"` | Unified version. The model loads the entire dataset onto the device, requiring sufficient device memory. Mini-batch processing is not supported. |
| `FastSeperatedSummitNMF` | `"fast_seperated"` or `"fs"` | Seperated version. The model loads the entire dataset onto the device, requiring sufficient device memory. Mini-batch processing is not supported. |

### Optimization

### Training a model

## 3. More examples and analysis

## 4. Citation
