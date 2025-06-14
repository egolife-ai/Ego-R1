# Ego-R1-Agent

A reinforcement learning framework for training reasoning-and-searching interleaved language models for egocentric video understanding. Built on [veRL](https://github.com/volcengine/verl) and [Search-R1](https://github.com/PeterGriffinJin/Search-R1?tab=readme-ov-file).

## 📁 Directory Structure

```
Ego-R1-Agent/
├── eval/                    # Evaluation scripts
│   ├── infer.py            # Main inference script
│   ├── infer_bench_summ.py # Benchmark evaluation
│   ├── infer_summ.py       # Summary evaluation
│   └── infer_bench_summ.sh # Benchmark evaluation script
├── utils/                   # Utility functions
│   ├── constants.py        # System prompts and API endpoints
│   ├── process.py          # Tool calling and response processing
│   └── serve.sh            # Model serving script
├── verl/                    # veRL framework components
├── ego_r1/                 # Core implementation
├── train_grpo.sh           # GRPO training script
├── train_grpo_base.sh      # Base GRPO training configuration
├── environment.yml         # Conda environment specification
├── setup.py               # Package setup
├── pyproject.toml         # Project metadata
└── VERL_README.md         # veRL framework documentation
```

## 🔧 Quick Setup

```bash
# Environment
conda create -n egor1 python=3.9
conda activate egor1
pip install -e .

# Training
bash train_grpo.sh

# Inference
bash utils/serve.sh
python eval/infer.py --model_name_or_path Ego-R1/Ego-R1-Agent-3B
```
