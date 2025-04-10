import os
import json
import csv
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import gc
import logging

# 配置日志
LOG_FILE = "/gemini/code/script.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# 配置
MODEL_PATH = "/gemini/code/DeepSeek-R1-Distill-Qwen-7B"
OUTPUT_DIR = "/gemini/code/output"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

# 加载模型和分词器
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16
    )
    model.to(DEVICE)
    logging.info(f"Model loaded on device: {DEVICE}")
except Exception as e:
    logging.error(f"模型加载失败: {e}")
    exit(1)

# 分割长文本为句子
def split_long_text(text, min_length=10):
    sentences = re.split(r'[.!?]\s+|\n+', text)
    split_texts = [sent.strip() + '.' for sent in sentences if len(sent.strip()) >= min_length]
    return split_texts if split_texts else [text]

# 筛选文本批次
def screen_text_batch(texts):
    if not texts:
        logging.warning("批次文本为空")
        return []
    
    prompt = """
以下是需要分类的文本。任务是将每个文本分类为 'positive'、'negative' 或 'neutral/insignificant'。
- 严格返回格式：["class1", "class2", ...]
- 禁止添加任何推理、说明或其他多余文本。
- 直接基于以下规则分类：
  - 'neutral/insignificant'：无意义的垃圾文本（如 '#@$%'、'ok'）或无情感、无意见的事实（如 '现在是下午3点'）。
  - 'positive'：包含积极情感、赞美、促销语气或表情符号（如 '好'、'棒'、'最佳'、'😊'、'🚀'）。
  - 'negative'：包含消极情感、批评或负面语气（如 '坏'、'糟糕'、'批评'、'😢'）。
  - 如有疑问，优先选择 'positive' 或 'negative'。
  - 注意：任何包含情感词、表情符号或语气的文本不得分类为 'neutral/insignificant'。

输入文本：
"""
    for i, text in enumerate(texts):
        prompt += f"Text {i+1}: \"{text}\"\n"
    prompt += "\n输出："

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.001,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    logging.info(f"筛选输入批次: {texts}")
    logging.info(f"原始筛选输出: {response}")
    
    match = re.search(r'\[(.*?)\]', response, re.DOTALL)
    cleaned_response = match.group(0) if match else response
    logging.info(f"清理后的筛选输出: {cleaned_response}")
    
    try:
        if match:
            classifications = [cls.strip(' "\',').replace('"', '') for cls in match.group(1).split(',')]
        else:
            if cleaned_response.startswith('[') and cleaned_response.endswith(']'):
                classifications = json.loads(cleaned_response)
            else:
                raise ValueError("输出格式无效")
        
        classifications = classifications[:len(texts)]
        if len(classifications) < len(texts):
            logging.warning(f"分类数量不足 ({len(classifications)}/{len(texts)})，用 'neutral/insignificant' 填充")
            classifications += ["neutral/insignificant"] * (len(texts) - len(classifications))
        
        for i, (text, cls) in enumerate(zip(texts, classifications)):
            if cls == "neutral/insignificant":
                if any(kw in text.lower() for kw in ['好', '棒', '最佳', 'exciting', 'amazing', '😊', '🚀']) or '!' in text:
                    classifications[i] = "positive"
                elif any(kw in text.lower() for kw in ['坏', '糟糕', 'risk', 'criticism', '😢']):
                    classifications[i] = "negative"
        
        logging.info(f"解析后的分类: {classifications}")
        return classifications
    except Exception as e:
        logging.error(f"解析筛选输出错误: {response}. 错误: {e}")
        return ["neutral/insignificant"] * len(texts)

# 情感分析批次
def analyze_sentiment_batch(texts, classifications):
    filtered_texts = [(text, cls) for text, cls in zip(texts, classifications) if cls in ["positive", "negative"]]
    if not filtered_texts:
        logging.warning("无需情感分析的文本")
        return []

    prompt = """
以下是需要情感分析的文本。任务是对每个标记为 'positive' 或 'negative' 的文本生成情感分析。
- 严格返回格式：[{"cleaned_text": "text", "sentiment": "positive/negative", "reasoning": "..."}, ...]
- 禁止添加任何多余文本，仅基于输入文本和分类生成结果。
- reasoning 需简洁，说明情感来源（如情感词、表情符号、语气）。

输入文本：
"""
    for i, (text, cls) in enumerate(filtered_texts):
        prompt += f"Text {i+1}: \"{text}\" Classification: \"{cls}\"\n"
    prompt += "\n输出："

    logging.info(f"情感分析输入: {filtered_texts}")
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.001,
            top_p=0.9,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip()
    logging.info(f"情感分析输出批次: {response}")
    
    match = re.search(r'\[.*\]', response, re.DOTALL)
    cleaned_response = match.group(0) if match else response
    logging.info(f"清理后的情感分析输出: {cleaned_response}")
    
    try:
        json_data = json.loads(cleaned_response)
        if len(json_data) != len(filtered_texts):
            logging.warning(f"情感分析输出数量 ({len(json_data)}) 与输入数量 ({len(filtered_texts)}) 不匹配")
        return json_data
    except Exception as e:
        logging.error(f"解析情感分析输出错误: {response}. 错误: {e}")
        return []

# 主处理函数
def process_file(file_path, batch_size=BATCH_SIZE):
    filename = os.path.basename(file_path).replace('.json', '')
    logging.info(f"处理文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'tweets' not in data:
        logging.error(f"{filename} 中没有 'tweets' 字段")
        return
    raw_tweets = [tweet.get('text', '') for tweet in data['tweets'] if 'text' in tweet]
    if not raw_tweets:
        logging.error(f"{filename} 中没有找到有效文本数据")
        return
    
    logging.info(f"原始推文数量: {len(raw_tweets)}")
    all_texts = []
    text_mapping = {}
    for i, tweet in enumerate(raw_tweets):
        split_texts = split_long_text(tweet)
        all_texts.extend(split_texts)
        text_mapping[i] = split_texts
    
    logging.info(f"分割后文本数量: {len(all_texts)}")
    total_batches = (len(all_texts) + batch_size - 1) // batch_size
    logging.info(f"总批次数: {total_batches}")
    
    all_sentiment_results = []
    all_csv_rows = []
    
    for batch_num in tqdm(range(total_batches), desc=f"处理 {filename}"):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, len(all_texts))
        batch_texts = all_texts[start_idx:end_idx]
        
        classifications = screen_text_batch(batch_texts)
        sentiment_results = analyze_sentiment_batch(batch_texts, classifications)
        
        logging.info(f"批次 {batch_num} 分类结果: {classifications}")
        logging.info(f"批次 {batch_num} 情感分析结果: {sentiment_results}")
        
        all_sentiment_results.extend(sentiment_results)
        
        for text, cls in zip(batch_texts, classifications):
            if cls in ["positive", "negative"]:
                all_csv_rows.append({"label": cls, "cleaned_text": text})
        
        torch.cuda.empty_cache() if DEVICE == "cuda" else None
        gc.collect()
    
    # 保存结果为 JSON 文件
    json_path = os.path.join(OUTPUT_DIR, f'cleaned_data_{filename}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_sentiment_results, f, ensure_ascii=False, indent=2)
    logging.info(f"JSON 文件保存至: {json_path}")
    
    # 保存结果为 CSV 文件
    if all_csv_rows:
        csv_path = os.path.join(OUTPUT_DIR, f'cleaned_data_{filename}.csv')
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['label', "cleaned_text"])
            writer.writeheader()
            writer.writerows(all_csv_rows)
        logging.info(f"CSV 文件保存至: {csv_path}")

# 主程序
def main():
    json_dir = "/gemini/code/json_files"
    json_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    if not json_files:
        logging.error("未找到 JSON 文件")
        return
    
    for json_file in tqdm(json_files, desc="处理文件"):
        file_path = os.path.join(json_dir, json_file)
        process_file(file_path)
        gc.collect()

if __name__ == "__main__":
    main()