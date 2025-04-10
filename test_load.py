import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logging.basicConfig(filename='/gemini/code/test_load.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("测试模型加载")

model_path = "/gemini/code/open_llama_7b"
try:
    logging.info("加载 tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    logging.info("tokenizer 加载完成")

    logging.info("加载模型到 GPU")
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",  # 直接加载到 GPU
        torch_dtype=torch.float16,
        quantization_config=quantization_config
    )
    logging.info("模型加载到 GPU 完成")
except Exception as e:
    logging.error(f"测试加载失败: {e}")
    raise

print("测试加载成功")