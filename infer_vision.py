from unsloth import FastVisionModel   # ← 必须第一行

from peft import PeftModel
from PIL import Image

BASE_MODEL = "Qwen/Qwen3.5-0.8B"
LORA_PATH = "D:/LLM/Qwen3.5-0.8b/output_qwen35_vision_lora"
TEST_IMAGE = "D:/LLM/Qwen3.5-0.8b/data/images/4.jpg"

def main():
    # 加载模型
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
    )
    
    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(model, LORA_PATH)
    
    # 切换到推理模式
    FastVisionModel.for_inference(model)

    # 准备图片
    image = Image.open(TEST_IMAGE).convert("RGB")
    
    # 构建对话格式
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "这图里的是谁"},
            ],
        }
    ]

    # 应用聊天模板
    text = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 处理输入（文本 + 图像）
    inputs = tokenizer(
        text=[text],
        images=[image],
        return_tensors="pt",
    ).to(model.device)

    # 生成输出
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,      # 最大生成长度
        temperature=0.2,         # 越低越确定
        top_p=0.9,               # 核采样
        do_sample=True,          # 开启采样
    )
    
    # 解码并打印结果
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)

if __name__ == "__main__":
    main()