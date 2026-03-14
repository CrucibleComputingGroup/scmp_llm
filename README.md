# LLM Full Attention

## Installation

```bash
conda create -n annstention python=3.10 -y
conda activate annstention
pip install torch torchvision torchaudio
pip install transformers datasets accelerate einops nvtx
pip install flash-attn --no-build-isolation
conda install pytorch::faiss-gpu
```

## Usage

Set `MODEL_PATH` in `test.py` to your local model path, then run:

```bash
python test.py
```
