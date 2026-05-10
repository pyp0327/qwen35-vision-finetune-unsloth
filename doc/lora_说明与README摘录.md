# LoRA（Low-Rank Adaptation，低秩适配）说明与项目摘录

## 1. LoRA 是什么

LoRA（Low-Rank Adaptation，低秩适配）是一种参数高效微调（PEFT）技术，专门用于在不大幅修改原模型的情况下，让模型学习新知识。

## 2. 核心原理（简单版）

- 大语言模型和视觉模型参数量很大（例如 Qwen3.5-0.8B 有数亿参数）。
- 全参数微调显存占用高，而且可能破坏原模型能力。
- LoRA 会在部分层（如 Attention、MLP）旁边插入小型低秩矩阵（A 和 B）。
- 训练时只更新这些小矩阵参数，原模型大部分权重保持冻结。
- 训练后通常只需保存 LoRA 适配器，不需要保存完整大模型。
- 推理时可将 LoRA 效果合并回模型，额外计算开销通常较小。

可以把 LoRA 理解为给模型加一件“轻量定制外套”，而不是重做整套参数。

## 3. 在你当前项目中的体现

训练脚本中的核心配置如下：

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,   # 同时微调视觉部分
    r=8,                           # LoRA rank（核心参数）
    lora_alpha=16,
    lora_dropout=0.0,
    ...
)
```

- `r=8`：LoRA 秩（rank）。值越大，可学习能力越强，但显存占用和过拟合风险也会增加。
- `finetune_vision_layers=True`：同时微调视觉模块，对“识别图片中的人物/对象”任务非常关键。
- 训练结束后主要保存 `adapter_model.safetensors` 等 LoRA 适配器文件。

## 4. LoRA 的优势

- 显存占用更低，8GB 显存也更容易跑通训练。
- 训练速度更快，配合 Unsloth 可获得明显加速。
- 模型切换方便，不同任务可切换不同 LoRA 文件。
- 更容易保留基础模型原有能力，降低灾难性遗忘风险。

## 5. 实际使用注意点

- 数据量很少时（例如仅 3 个样本）容易过拟合，可能“记住训练图”但泛化差。
- 多任务或多人物场景，建议将新数据追加到同一个 `train.jsonl` 中继续训练。
- 可尝试的调参方向：
  - `r`：8 -> 16
  - `lora_dropout`：0.05 ~ 0.1
  - `learning_rate`：适当降低

## 6. README_finetune_vision.md 相关摘录（LoRA 部分）

以下内容整理自 `D:/LLM/Qwen3.5-0.8b/README_finetune_vision.md` 的 LoRA 相关章节：

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,      # 微调视觉部分（非常重要）
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8,                              # LoRA rank
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    random_state=3407,
)
```

README 中对应说明要点：

- `FastVisionModel.get_peft_model(...)` 用于给基础模型添加 LoRA 适配器（只训练少量参数）。
- `finetune_vision_layers=True` 可提升图片理解能力。
- `r=8` 可按显存和效果调整；更大通常能力更强，但过拟合风险更高。
- 小数据集训练时，应重点关注过拟合并结合新图片进行验证。
