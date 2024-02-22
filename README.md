
# GCAN

We are excited to introduce GCAN, our latest method implemented in PyTorch, leveraging the power of RTXA4000 GPUs. This approach, detailed in our [published work](https://www.sciencedirect.com/science/article/abs/pii/S0950705124001448?via%3Dihub), is designed to enhance session-based recommendations through graph-enhanced and collaborative attention networks.

## Getting Started

To utilize GCAN, begin by generating the collaborative similarity file:

1. Execute `collaborativeSimilarity.py`:

   ```bash
   python collaborativeSimilarity.py --dataset=RetailRocket
   ```
2. Proceed to train the model with `src/main.py`:

   ```bash
   python src/main.py --dataset=RetailRocket
   ```

## Prerequisites

Ensure the following dependencies are installed:

- Python 3.7
- PyTorch 1.11
- NumPy 1.21.5
- NetworkX 2.2

## Citing GCAN

If you find GCAN useful in your research, please consider citing our work:

```bibtex
@article{zhu2024graph,
  title={Graph-enhanced and collaborative attention networks for session-based recommendation},
  author={Zhu, Xiaoyan and Zhang, Yu and Wang, Jiayin and Wang, Guangtao},
  journal={Knowledge-Based Systems},
  pages={111509},
  year={2024},
  publisher={Elsevier}
}
```
