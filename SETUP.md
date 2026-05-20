# SkillFlow Agent - 部署指南

## 1. 环境安装

```bash
# 创建 conda 环境
conda create -n vllm-prompt python=3.10 -y
conda activate vllm-prompt

# 安装依赖
pip install -r requirements.txt
```

关键依赖：
- `vllm` (推理服务)
- `peft` (LoRA 训练)
- `transformers` (模型加载)
- `openai` (API 客户端)
- `sentence-transformers` (语义搜索 bge-base-en-v1.5)
- `torch` (训练)

## 2. 模型准备

需要两个模型：
- **Supervisor**: Qwen3-8B (可训练，带 θ-LoRA)
- **M_exec**: gpt-oss-120b 或其他大模型 (冻结执行器)

```bash
# 下载 Qwen3-8B（如果 HF cache 没有）
huggingface-cli download Qwen/Qwen3-8B

# 下载 bge-base-en-v1.5（语义搜索用）
huggingface-cli download BAAI/bge-base-en-v1.5
```

## 3. 启动 vLLM 服务

```bash
export SGLANG_API_KEY="<LOCAL_API_KEY>"  # 本地服务用；不要提交真实密钥

# Supervisor (Qwen3-8B + LoRA)
CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve Qwen3-8B \
  --port 8005 \
  --max-model-len 16384 \
  --enable-lora \
  --max-lora-rank 64 \
  --api-key "$SGLANG_API_KEY"

# M_exec (大模型执行器) - 在另一个 GPU 上
CUDA_VISIBLE_DEVICES=<GPU_ID> vllm serve <model-name> \
  --port 8010 \
  --max-model-len 16384 \
  --api-key "$SGLANG_API_KEY"
```

## 4. 修改配置

编辑 `configs/skillflow.yaml`：
```yaml
supervisor_api_base: "http://localhost:8005/v1"
supervisor_model: "supervisor_theta"
executor_api_base: "http://localhost:8010/v1"
executor_model: "<your-model-name>"
```

## 5. 启动训练

```bash
TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=<GPU_ID> \
  python -u run_training.py \
  --config configs/skillflow.yaml \
  --max-steps 300 \
  --gpu <GPU_ID> \
  --fresh
```

## 6. 数据说明

- `data/train_v3.json`: 主训练集（按需生成/下载，不建议直接提交大文件）
- `data/test_iid_v3.json`: IID 测试集
- `data/FlowSteer-Dataset/`: 评估基准 (AIME, HumanEval, MATH, HotpotQA, MuSiQue 等)

## 7. GPU 分配

最少需要 2 张 GPU：
- GPU A: Supervisor vLLM (Qwen3-8B, ~16GB) + 训练 (LoRA, ~8GB) = ~24GB
- GPU B: M_exec vLLM (大模型, 按模型大小)

推荐 3 张：
- GPU 1: 训练 (LoRA backward)
- GPU 2: Supervisor vLLM
- GPU 3: M_exec vLLM
