from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator

from datasets import load_dataset
from trl import SFTConfig, SFTTrainer   # trl 放在后面


MODEL_NAME = "Qwen/Qwen3.5-0.8B"
DATA_PATH = "D:/LLM/Qwen3.5-0.8b/data/train.jsonl"
OUTPUT_DIR = "D:/LLM/Qwen3.5-0.8b/output_qwen35_vision_lora"


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


def main():
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=MODEL_NAME,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
        max_seq_length=2048,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    ds = load_dataset("json", data_files=DATA_PATH, split="train")
    ds = ds.map(to_messages, remove_columns=ds.column_names)

    sft_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # 或调到 8 以降低显存压力
        learning_rate=5e-5,              # 建议适当降低，防止震荡或过拟合
        num_train_epochs=50,             # ← 这里改成 50
        warmup_steps=5,                  # 适当增加 warmup
        logging_steps=1,
        save_steps=20,                   # 增加保存间隔，避免保存太多 checkpoint
        save_total_limit=3,
        remove_unused_columns=False,
        dataset_text_field=None,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=1024,                 # 建议先降到 1024，加快训练
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        report_to="none",
        packing=False,
        eos_token=None,
        pad_token=None,
        weight_decay=0.01,               # 新增：增加权重衰减，减轻过拟合
        lr_scheduler_type="cosine",      # 新增：余弦学习率调度，更平稳
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,       # 用 processing_class 而非 tokenizer
        train_dataset=ds,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        args=sft_args,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"训练完成，LoRA 已保存到: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
