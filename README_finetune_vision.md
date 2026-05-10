# Qwen3.5-0.8B 三图图片微调（Unsloth）

## 1. 放图片
- 把你的 3 张图片放到 `D:/LLM/Qwen3.5-0.8b/data/images/`
- 文件名按 `1.jpg`, `2.jpg`, `3.jpg`（或你自己改 `train.jsonl` 里的路径）

## 2. 填训练标注
- 打开 `D:/LLM/Qwen3.5-0.8b/data/train.jsonl`
- 把每条里的 `answer` 改成你希望模型学习的中文描述

## 3. 开始训练
```powershell
conda activate unsloth_env
cd D:\LLM\Qwen3.5-0.8b
python .\finetune_vision.py
```

## 4. 训练后推理测试
```powershell
conda activate unsloth_env
cd D:\LLM\Qwen3.5-0.8b
python .\infer_vision.py
```

## 5. 注意
- 只有 3 张图很容易过拟合，这是正常现象。
- 先跑通流程，再逐步加到 20~50 条会稳定很多。
