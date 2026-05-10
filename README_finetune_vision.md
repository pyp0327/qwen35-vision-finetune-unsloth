**Qwen3.5-0.8B Vision LoRA 微调与推理完整新手指南（最终版）**

**📌 目标**：用最细致的方式，把**训练脚本**和**推理脚本**完整逐行拆解。适合之前只用 Agent + API 的用户过渡到本地微调。

---

### **1. 整体概念**

* **训练阶段**：使用你的图片+文字数据，对基础模型添加 LoRA 适配器进行微调，让模型学会识别特定视觉内容（如人物身份）。
* **推理阶段**：加载基础模型 + LoRA 权重，输入新图片和问题，生成回答。
* **核心区别于 API**：API 高度封装，这里需要手动完成加载、输入准备和生成，但可控性更强（可调整量化、LoRA 参数等）。

---

### **2. 训练脚本完整逐行拆解**

```python
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer   # trl 放在后面
```

* `from unsloth import ...`：**必须放在脚本最顶部**。Unsloth 会自动 patch（修改）后续库，实现加速和优化。如果顺序错误，加速功能会失效。
* **FastVisionModel**：Unsloth 专门为视觉多模态模型提供的加载和训练类。
* **is_bf16_supported**：检查你的 GPU 是否支持 bf16 精度（更快、更省显存）。
* **UnslothVisionDataCollator**：专门处理“图片 + 文字”混合数据的批处理工具。
* **load_dataset**：用于读取 JSONL 训练文件。
* **SFTConfig / SFTTrainer**：来自 TRL 库，用于监督微调的配置和执行器。

---

```python
MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DATA_PATH = "D:/LLM/Qwen3.5-0.8b/data/train.jsonl"
OUTPUT_DIR = "D:/LLM/Qwen3.5-0.8b/output_qwen35_vision_lora"
```

* **MODEL_NAME**：基础模型的 Hugging Face 标识符，会自动从本地缓存（C:\Users\你的用户名.cache\huggingface\hub）查找。
* **DATA_PATH**：训练数据文件路径（JSONL 格式，每行包含 image、instruction、answer 字段）。
* **OUTPUT_DIR**：最终 LoRA 权重保存路径，同时会自动生成 checkpoint-xxx 中间文件夹。

---

```python
class SafeSFTTrainer(SFTTrainer):
    # Work around a TRL+VLM eos_token validation bug that can inject "<EOS_TOKEN>".
    def __init__(self, *args, **kwargs):
        cfg = kwargs.get("args")
        if cfg is None:
            raise RuntimeError("SafeSFTTrainer: 没有收到 args=...，无法安全覆盖 eos_token。")
        cfg.eos_token = None
        cfg.pad_token = None
        kwargs["args"] = cfg
        super().__init__(*args, **kwargs)
```

* 这是一个**自定义安全包装类**，继承自 SFTTrainer。
* **作用**：绕过 TRL 在视觉语言模型上对 eos_token 的严格校验问题（Qwen 系列常见）。
* **使用建议**：先尝试普通 SFTTrainer，如果报 eos_token 相关错误，再替换为 SafeSFTTrainer。

---

```python
def to_messages(example):
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": example["image"]},
                    {"type": "text", "text": example["instruction"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["answer"]}],
            },
        ]
    }
```

* **数据格式转换函数**（核心之一）。
* 把 JSONL 中的每条样本转为标准的**多模态对话格式**（messages 列表）。
* `role: "user"` 包含图片和指令（输入部分）。
* `role: "assistant"` 是期望答案（模型学习的目标）。
* **关键**：instruction 文字应尽量与后续推理 prompt 保持一致。

---

```python
def main():
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=2048,
    )
```

* **FastVisionModel.from_pretrained**：加载基础模型。
* `model_name=MODEL_NAME`：指定模型标识。
* `load_in_4bit=True`：4bit 量化加载，显著降低显存占用。
* `use_gradient_checkpointing="unsloth"`：Unsloth 优化的梯度检查点，节省显存。
* `max_seq_length=2048`：最大序列长度（图片 token + 文本）。
* 返回 `model`（模型本体）和 `tokenizer`（处理器）。

---

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

* **FastVisionModel.get_peft_model**：添加 LoRA 适配器（只训练少量参数）。
* `finetune_vision_layers=True`：同时微调视觉编码器，提升图片理解能力。
* `r=8`：LoRA 秩，可根据显存和效果调整（更大则能力更强，但过拟合风险更高）。

---

```python
    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(to_messages, remove_columns=ds.column_names)
```

* `load_dataset("json", ...)`：读取 JSONL 文件。
* `ds.map(...)`：对每条数据应用格式转换函数。
* `remove_columns`：清理不需要的列，节省内存。

---

```python
    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        num_train_epochs=50,          # 当前设置的轮数
        warmup_steps=5,
        logging_steps=1,
        save_steps=20,
        save_total_limit=2,
        remove_unused_columns=False,
        dataset_text_field=None,     # 多模态必须为 None
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=2048,
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        report_to="none",
        packing=False,
        eos_token=None,
        pad_token=None,
        # 可额外添加：weight_decay=0.01, lr_scheduler_type="cosine"
    )
```

* **SFTConfig**：所有训练超参数的集中配置。
* `per_device_train_batch_size=1 + gradient_accumulation_steps=4`：有效 batch size=4，适合小显存。
* `num_train_epochs=50`：训练轮数（小数据集下需注意过拟合）。
* `dataset_text_field=None`：多模态场景必须关闭自动文本处理。

---

```python
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,       # Vision 模型推荐使用 processing_class
        train_dataset=ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=sft_args,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成，LoRA 已保存到: {OUTPUT_DIR}")
```

* **SFTTrainer**：创建并执行训练器。
* `data_collator=UnslothVisionDataCollator(...)`：负责图像和文本的批处理。
* `trainer.train()`：启动训练循环。
* 训练结束后保存最终 LoRA 权重和 tokenizer 到 OUTPUT_DIR。

---

### **3. 推理脚本完整逐行拆解**

```python
def main():
    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
    )
```

* `FastVisionModel.from_pretrained`：Unsloth 提供的加载函数。
* `model_name=BASE_MODEL`：指定基础模型。
* `load_in_4bit=True`：4bit 量化加载。
* 返回 model 和 tokenizer。

```python
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, LORA_PATH)
```

* `PeftModel.from_pretrained`：加载训练好的 LoRA 适配器。
* LORA_PATH 是保存的文件夹路径。

```python
    # 切换到推理模式
    FastVisionModel.for_inference(model)
```

* 优化推理速度和显存，必须调用。

```python
    # 准备图片
    image = Image.open(TEST_IMAGE).convert("RGB")
```

* 用 PIL 打开并统一转为 RGB 格式。

```python
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "这图里的是谁，叫什么名字"},
            ],
        }
    ]
```

* 多模态输入模板，可修改 text 中的 prompt。

```python
    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
```

* 转为字符串，添加生成提示。

```python
    # 处理输入（文本 + 图像）
    inputs = tokenizer(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)
```

* 生成 PyTorch 张量并搬到 GPU。

```python
    # 生成输出
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,      
        temperature=0.2,         
        top_p=0.9,               
        do_sample=True,          
    )
```

* 核心生成函数，参数控制长度和随机性。

```python
    # 解码并打印结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
```

* 将 token 转回文字并输出。

---

### **4. 训练新人物的处理建议**

* **推荐方式**：**直接把新人物的数据追加到现有 train.jsonl 文件中**，继续使用同一个 OUTPUT_DIR 训练。
* 优点：模型能学会区分不同人物，效果通常更好。
* **不需要清空训练集或删除 output**（除非想完全重新开始）。
* 训练后用不同人物的**全新图片**分别测试。

---

### **5. 完整使用流程**

1. 准备/追加 JSONL 数据。
2. 运行训练脚本。
3. 运行推理脚本（定义好 BASE_MODEL、LORA_PATH、TEST_IMAGE 等变量）。
4. 观察 loss 和新图片测试结果，必要时调整参数或增加数据。

这个版本已覆盖所有关键部分。如需**两个完整可直接复制的 .py 文件内容**（含所有 import 和 if **name** == "**main**"），或 JSONL 数据模板，请直接说明。
