# MoRAM — X-TAIL Benchmark (CLIP Continual Learning)

This directory contains the code for evaluating **MoRAM** on the **X-TAIL** benchmark: continual few-shot adaptation of CLIP across 10 image classification domains.

> Back to [main README](../README.md)

---

## Environment Setup

```bash
conda create -n moram_clip python=3.12 -y
conda activate moram_clip

# Install PyTorch matching your CUDA version, e.g.:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```

## Data Preparation

The X-TAIL benchmark uses **10 classification datasets** in the following default task sequence:

1. Aircraft
2. Caltech-101
3. DTD
4. EuroSAT
5. Oxford Flowers
6. Food-101
7. MNIST
8. Oxford Pets
9. Stanford Cars
10. SUN397

Download and organize all datasets under a single root directory. Follow the dataset preparation guide from [CoOp DATASETS.md](https://github.com/KaiyangZhou/CoOp/blob/main/DATASETS.md).

Your data directory should look like:

```
<data_dir>/
├── fgvc_aircraft/
├── caltech-101/
├── dtd/
├── eurosat/
├── oxford_flowers/
├── food-101/
├── mnist/
├── oxford_pets/
├── stanford_cars/
└── sun397/
```

## Running MoRAM

### Quick Start

From this directory, with your Python environment activated and datasets under `XTAIL_DATA_DIR` (default `./datasets`):

```bash
cd XTAIL
bash runner_moram.sh
```

## Acknowledgement

This benchmark builds on [MoE-Adapters](https://github.com/JiazuoYu/MoE-Adapters4CL), [RAIL](https://github.com/linghan1997/Regression-based-Analytic-Incremental-Learning), and [CoDyRA](https://github.com/jeff024/codyra).
