# PuzzleTensor
This repository is the official implementation of "PuzzleTensor: A Method-Agnostic Data Transformation for Compact Tensor Factorization" (KDD 2025).
<p align="center">
  <img src="https://raw.githubusercontent.com/snudatalab/PuzzleTensor/main/docs/ex-0.gif" width="320"/>
  <img src="https://raw.githubusercontent.com/snudatalab/PuzzleTensor/main/docs/ex-1.gif" width="320"/>
  <br>
  <img src="https://raw.githubusercontent.com/snudatalab/PuzzleTensor/main/docs/mnist-0.gif" width="320"/>
  <img src="https://raw.githubusercontent.com/snudatalab/PuzzleTensor/main/docs/mnist-1.gif" width="320"/>
</p>

## Abstract
How can we achieve compact tensor representations without sacrificing reconstruction accuracy? Tensor decomposition is a cornerstone of modern data mining and machine learning, enabling efficient representations of multi-dimensional data through fundamental algorithms such as CP, Tucker, and Tensor-Train decompositions. However, directly applying these methods to raw data often results in high target ranks, poor reconstruction accuracy, and computational inefficiencies, as the data may not naturally conform to the low-rank structures these methods assume.
In this paper, we propose **PuzzleTensor**, a method-agnostic data transformation technique for compact tensor factorization. Given a data tensor, PuzzleTensor "solves the puzzle" by shifting each hyperslice of the tensor to achieve accurate decompositions with significantly lower target ranks. PuzzleTensor offers three key advantages: (1) it is independent of specific decomposition methods, making it seamlessly compatible with various algorithms, such as CP, Tucker, and Tensor-Train decompositions; (2) it works under weak data assumptions, showing robust performance across both sparse and dense data, regardless of the rank; (3) it is inherently explainable, allowing clear interpretation of its learnable parameters and layer-wise operations. Extensive experiments show that PuzzleTensor consistently outperforms direct tensor decomposition approaches by achieving lower reconstruction errors and reducing the required target rank, making it a versatile and practical tool for compact tensor factorization in real-world applications.



## Prerequisites
- `numpy==1.26.4`
- `tensorly==0.8.1`
- `torch==2.3.0`


## Datasets
We provide the synthetic and real-world datasets used in our experiments below.

| Dataset | Type | Size | Density | Source |
|---------|------|------|---------|--------|
| $D_{n=4,\cdots,8}$ | Synthetic | $2^n \times 2^n \times 2^n$ | $1.000$ | [LINK](https://drive.google.com/open?id=1fkwuug02bgqnRTVNWvSI3bk9Ks1i0DQF&usp=drive_copy) | 
| $S_{n=4,\cdots,8}$ | Synthetic | $2^n \times 2^n \times 2^n$ | $0.010$ | [LINK](https://drive.google.com/open?id=1fkwuug02bgqnRTVNWvSI3bk9Ks1i0DQF&usp=drive_copy) | 
| Uber | Real-world | $183 \times 24 \times 1140$ | $0.138$ | [LINK](http://frostt.io/) | 
| Action | Real-world | $100 \times 570 \times 567$ | $0.393$ | [LINK](https://github.com/titu1994/MLSTM-FCN) | 
| PEMS-SF | Real-world | $963 \times 144 \times 440$ | $0.999$ | [LINK](https://www.timeseriesclassification.com/) | 
| Activity | Real-world | $337 \times 570 \times 320$ | $0.569$ | [LINK](https://github.com/titu1994/MLSTM-FCN) | 
| Stock | Real-world | $1317 \times 88 \times 916$ | $0.816$ | [LINK](https://github.com/jungijang/KoreaStockData) | 
| NYC | Real-world | $265 \times 265 \times 28 \times 35$ | $0.118$ | [LINK](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) | 


## Reference

If you use this code, please cite the following paper.
```bibtex
@inproceedings{ParkKK25,
  title     = {PuzzleTensor: A Method-Agnostic Data Transformation for Compact Tensor Factorization},
  author    = {Park, Yong-chan and Kim, Kisoo and Kang, U},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD ’25)},
  year      = {2025}
}
