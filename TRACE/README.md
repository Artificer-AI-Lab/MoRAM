# MoRAM — TRACE Benchmark (LLM Continual Learning)

This directory contains the code for evaluating **MoRAM** on the **TRACE** benchmark: continual learning for Large Language Models across 8 diverse NLP tasks using DeepSpeed.

> Back to [main README](../README.md)

---

## Environment Setup

```bash
conda create -n moram_llm python=3.10 -y
conda activate moram_llm

# Install PyTorch matching your CUDA version, e.g.:
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
pip install -r requirements.txt
```



## Data Preparation

The TRACE benchmark data is included in this repository under `data/LLM-CL-Benchmark/LLM-CL-Benchmark_500/`.

The default 8-task continual learning sequence:

| # | Task | Type | Metric |
|:--|:-----|:-----|:-------|
| 1 | C-STANCE | Stance detection | Accuracy |
| 2 | FOMC | Sentiment classification | Accuracy |
| 3 | MeetingBank | Summarization | ROUGE-L |
| 4 | Py150 | Code completion | Similarity |
| 5 | ScienceQA | Question answering | Accuracy |
| 6 | NumGLUE-cm | Math (commonsense) | Accuracy |
| 7 | NumGLUE-ds | Math (data science) | Accuracy |
| 8 | 20Minuten | Text simplification | SARI |


## Model Preparation

Download a pretrained model from HuggingFace. Supported architectures include LLaMA, Gemma, and others.

```bash
# Example: download Gemma-2B-it
mkdir -p PTM && cd PTM
git clone https://huggingface.co/google/Gemma-2B-it
cd ..
```

Or simply pass a HuggingFace model identifier (e.g., `google/Gemma-2B-it`) and let the library download it automatically.

## Running MoRAM

The full pipeline has three stages: **training**, **inference**, and **metric collection**. The provided `runner_moram.sh` orchestrates all three.

### Quick Start

From this directory, with your Python environment activated:

```bash
cd TRACE
bash runner_moram.sh
```

## Acknowledgement

This benchmark builds on [TreeLoRA](https://github.com/QianYuanYZ/TreeLoRA) and [TRACE](https://github.com/BeyonderXX/TRACE).
