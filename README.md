# LightGCN-PyTorch (Optimized Implementation)

This repository provides an optimized PyTorch implementation of **LightGCN**, tailored for robustness, cross-platform hardware acceleration, and industrial deployment.

**Attribution:**
This project is a secondary development based on the original academic implementation by [Jianbai Ye](https://github.com/gusye1234/LightGCN-PyTorch) for the SIGIR 2020 paper: *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation*.

## Key Improvements

This fork introduces several engineering and reliability enhancements over the original academic codebase (Modified by @chenz0zz0):

- **Decoupled Inference API (`predict.py`)**: Added a standalone script for Top-K item retrieval. It includes business-logic filtering to automatically exclude positively interacted items, simulating standard industrial vector retrieval.
- **Apple Silicon (MPS) Compatibility**: Resolved sparse matrix operation failures on Apple M-series chips by implementing intelligent CPU fallback mechanisms for incompatible tensor operations, while retaining GPU/MPS acceleration for dense neural network computations.
- **Dynamic Early Stopping**: Replaced fixed-epoch training with a patience-based early stopping mechanism monitored via `Recall@20`, including automated saving of the optimal model weights.
- **Robust Data Parsing**: Enhanced the `dataloader.py` to gracefully handle malformed datasets (e.g., irregular spacing, trailing whitespaces, and "ghost users" without item interactions), specifically resolving parsing crashes on the `amazon-book` dataset.
- **Type Stability & API Modernization**: 
  - Fixed `TypeError` issues associated with length-1 array unpacking during evaluation label generation.
  - Upgraded legacy PyTorch sparse matrix initializations to the modern `torch.sparse_coo_tensor` API.
  - Implemented native Python sampling to eliminate C++ compilation dependencies across different operating systems.

## Supported Datasets

The dataloader has been tested and verified with the following benchmark datasets:
- `gowalla`
- `yelp2018`
- `amazon-book`
- `lastfm`

*(Raw data files should be placed inside the `data/` directory. If using custom datasets, follow the same format as benchmark datasets: `user_id item_id [rating (optional)] timestamp (optional)`)*.

## Usage

### 1. Installation
Install required packages with:
```bash
pip install -r requirements.txt
```
- Python >= 3.8
- PyTorch >= 1.10

---

### 2. Data Preparation
This repository supports all four standard benchmark datasets from the original LightGCN paper.  
All datasets are included in the `data/` directory, ready to use:
- `gowalla`
- `yelp2018`
- `amazon-book`
- `lastfm`

The `dataloader.py` automatically handles:
- Cleaning invalid lines and malformed entries
- Mapping user/item IDs to continuous indices
- Building the normalized graph adjacency matrix

### 3. Training the Model
```bash
cd code
python main.py --dataset gowalla --layer 3 --recdim 64 --lr 0.001 --decay 1e-4
```

#### Key Parameters
| Parameter       | Description                                                                 | Default |
|-----------------|-----------------------------------------------------------------------------|---------|
| `--dataset`     | Dataset name (gowalla/yelp2018/amazon-book/lastfm)                          | gowalla |
| `--layer`       | Number of LightGCN layers                                                   | 3       |
| `--recdim`      | Embedding dimension                                                         | 64      |
| `--lr`          | Learning rate                                                               | 0.001   |
| `--decay`       | L2 regularization weight                                                    | 1e-4    |

### 4. Inference & Recommendation
Run the standalone inference script to get top-K recommendations for a specific user.
You can specify any supported dataset:
```bash
python predict.py --dataset gowalla
```
- Automatically excludes items the user has already interacted with
- Outputs a list of recommended item IDs

## Performance
This implementation maintains consistent performance with the original LightGCN model across all supported datasets.

## License
This project is based on the original LightGCN-PyTorch implementation, licensed under the MIT License.

- Original work: Copyright (c) 2020 Jianbai Ye
- Modified work: Copyright (c) chenz0zz0

See the `LICENSE` file for full details.

## Acknowledgements
- Original implementation: [gusye1234/LightGCN-PyTorch](https://github.com/gusye1234/LightGCN-PyTorch)
- Paper: *LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation* (SIGIR 2020)
- Authors: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, Meng Wang
