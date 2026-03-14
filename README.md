# LLM Full Attention

基于 LLaMA 架构的全量注意力（Full Attention）研究项目，探索在大语言模型推理中使用近似最近邻搜索（ANNS）加速注意力计算的方法。

## 项目结构

```
.
├── model/
│   └── llama_anns.py     # 修改后的 LLaMA 模型，集成 NVTX profiling 标记
├── __init__.py
├── Dockerfile            # CUDA 12.4 + RAPIDS + FAISS 环境
└── requirements.txt
```

## 环境依赖

- CUDA 12.4
- PyTorch 2.5.1
- Transformers 4.51.3
- FAISS-GPU（via conda `pytorch::faiss-gpu`）
- RAPIDS（cuML, cuGraph 等）
- Flash Attention 2

### Docker 构建

```bash
docker build -t llm-full-attention .
```

Dockerfile 基于 `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`，自动配置 Miniconda 环境 `annstention`，并安装所有依赖。

### 手动安装

```bash
conda create -n annstention python=3.10 -y
conda activate annstention
pip install torch torchvision torchaudio
pip install transformers datasets accelerate
pip install flash-attn --no-build-isolation
conda install pytorch::faiss-gpu
```

## 核心模块

### `model/llama_anns.py`

基于 HuggingFace Transformers 的 LLaMA 实现，主要修改包括：

- 在 `LlamaDecoderLayer.forward` 中添加了 CUDA NVTX profiling 标记，用于精细化分析各子模块（self-attention、residual、MLP）的 GPU 耗时
- 在 `LlamaModel.forward` 中记录各层输入的隐藏状态（`layer_inputs`），支持层间相似度分析
- 支持标准 LLaMA 的所有任务头：CausalLM、SequenceClassification、QuestionAnswering、TokenClassification

### NVTX Profiling

代码中已集成 `torch.cuda.nvtx` 标记，可配合 Nsight Systems 进行 GPU 性能分析：

```bash
nsys profile -o profile_output python your_script.py
```

## 使用方法

```python
from model.llama_anns import LlamaForCausalLM
from transformers import LlamaConfig

config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", config=config)
```

## 参考

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [FAISS](https://github.com/facebookresearch/faiss)
