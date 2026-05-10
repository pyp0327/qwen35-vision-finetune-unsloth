# check_vision_model.py
from unsloth import FastVisionModel

model_name = "Qwen/Qwen3.5-0.8B"

model, tokenizer = FastVisionModel.from_pretrained(
    model_name=model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)
print("OK: 可按视觉模型加载")