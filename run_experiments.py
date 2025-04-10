import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
import logging
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments
from peft import PeftModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import time
import json
import os
import psutil
import gc
from datasets import Dataset
import unicodedata

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/gemini/code/experiment.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 监控显存和系统内存（不输出）
def log_memory_usage():
    pass  # 不输出内存信息

# 加载文化词典
def load_cultural_dict(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        cultural_dict = json.load(f)
    return cultural_dict

# 清洗文本中的特殊字符
def clean_text(text):
    # 将 Unicode 转义字符转换为实际字符
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    # 替换常见转义字符
    text = text.replace('\u2019', "'").replace('\u002c', ',')
    return text

# 加载数据集
def load_data(file_path, max_samples=500, is_train=False):
    data = pd.read_csv(file_path)
    # 检查空值
    logger.info(f"数据集 {file_path} 空值检查: {data.isnull().sum()}")
    data = data.dropna(subset=['text'])  # 统一使用 'text' 列
    data['text'] = data['text'].apply(clean_text)
    # 确保标签为整数
    data['label'] = data['label'].astype(int)
    # 检查标签值是否合法（0 或 1）
    if not data['label'].isin([0, 1]).all():
        logger.error(f"数据集 {file_path} 包含非法标签值: {data['label'].unique()}")
        raise ValueError(f"数据集 {file_path} 包含非法标签值")
    texts = data["text"].tolist()
    labels = data["label"].values
    enhanced_texts = None
    
    # 检查是否存在预翻译的 zh_translated_text 和 back_translated_text 列
    if not is_train and 'zh_translated_text' in data.columns and 'back_translated_text' in data.columns:
        zh_translated_texts = data["zh_translated_text"].tolist()
        back_translated_texts = data["back_translated_text"].tolist()
        # 检查是否为空
        if all(pd.isna(text) for text in zh_translated_texts) and all(pd.isna(text) for text in back_translated_texts):
            enhanced_texts = None
            logger.info("zh_translated_text 和 back_translated_text 列为空，将进行实时翻译增强")
        else:
            enhanced_texts = back_translated_texts
            logger.info("加载预翻译的 back_translated_text 列")
    else:
        enhanced_texts = None
        logger.info("未找到 zh_translated_text 和 back_translated_text 列，将进行实时翻译增强")
    
    # 限制样本数量以减少内存压力
    if len(data) > max_samples:
        data = data.sample(n=max_samples, random_state=42)
        texts = data["text"].tolist()
        labels = data["label"].values
        if not is_train and enhanced_texts is not None:
            enhanced_texts = data["back_translated_text"].tolist()
    
    logger.info(f"{'训练集' if is_train else '测试集'}标签分布: {pd.Series(labels).value_counts().to_dict()}")
    return texts, labels, enhanced_texts

# 加载模型和分词器
def load_models_and_tokenizers():
    # 加载 XLM-RoBERTa
    logger.info("加载 XLM-RoBERTa 模型...")
    xlm_path = "/gemini/code/finetuned_xlm_roberta"
    if not os.path.exists(xlm_path):
        logger.error(f"XLM-RoBERTa 模型路径 {xlm_path} 不存在！")
        raise FileNotFoundError(f"XLM-RoBERTa 模型路径 {xlm_path} 不存在！")
    logger.info(f"XLM-RoBERTa 模型路径 {xlm_path} 存在，包含文件: {os.listdir(xlm_path)}")
    xlm_model = AutoModelForSequenceClassification.from_pretrained(xlm_path)
    xlm_tokenizer = AutoTokenizer.from_pretrained(xlm_path)
    xlm_model.to(device)
    log_memory_usage()
    torch.cuda.empty_cache()

    # 加载 DeepSeek（使用 DeepSeek-7B 模型，启用量化）
    logger.info("加载 DeepSeek 模型...")
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            "/gemini/code/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True,
            num_labels=2,
            quantization_config=quantization_config
        )
        # 验证 PEFT 模型路径
        peft_path = "/gemini/code/instruction_finetuned_deepseek"
        if not os.path.exists(peft_path):
            logger.error(f"PEFT 模型路径 {peft_path} 不存在！")
            raise FileNotFoundError(f"PEFT 模型路径 {peft_path} 不存在！")
        logger.info(f"PEFT 模型路径 {peft_path} 存在，包含文件: {os.listdir(peft_path)}")
        
        deepseek_model = PeftModel.from_pretrained(
            base_model,
            peft_path,
            trust_remote_code=True
        )
        deepseek_tokenizer = AutoTokenizer.from_pretrained(
            "/gemini/code/DeepSeek-R1-Distill-Qwen-7B",
            trust_remote_code=True
        )
        logger.info(f"加载 PEFT 模型后的结构: {deepseek_model}")
    except Exception as e:
        logger.error(f"无法加载 DeepSeek 模型: {e}")
        raise

    # 设置 DeepSeek 分词器的 pad_token
    if deepseek_tokenizer.pad_token is None:
        deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token
        deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id
        logger.info("为 DeepSeek 分词器设置 pad_token: %s", deepseek_tokenizer.pad_token)

    if deepseek_tokenizer.pad_token is None:
        deepseek_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        deepseek_tokenizer.pad_token_id = deepseek_tokenizer.convert_tokens_to_ids('<pad>')
        deepseek_model.resize_token_embeddings(len(deepseek_tokenizer))
        logger.info("手动为 DeepSeek 分词器添加 pad_token: <pad>")

    if not hasattr(deepseek_model.config, 'pad_token_id') or deepseek_model.config.pad_token_id is None:
        deepseek_model.config.pad_token_id = deepseek_tokenizer.pad_token_id
        logger.info("为 DeepSeek 模型设置 pad_token_id: %d", deepseek_model.config.pad_token_id)

    # 将 DeepSeek 模型加载到 GPU 上，加速推理
    deepseek_model.to(device)
    log_memory_usage()
    torch.cuda.empty_cache()

    models = {
        "xlm": xlm_model,
        "deepseek": deepseek_model
    }
    tokenizers = {
        "xlm": xlm_tokenizer,
        "deepseek": deepseek_tokenizer
    }

    return models, tokenizers

# 延迟加载 mBART 模型
def load_m2m_model():
    m2m_model_path = "/gemini/code/models/mbart_finetuned_v5"
    logger.info("加载 mBART 模型...")
    try:
        m2m_tokenizer = AutoTokenizer.from_pretrained(m2m_model_path)
        m2m_model = AutoModelForSeq2SeqLM.from_pretrained(m2m_model_path)
        logger.info(f"成功加载 mBART 模型: {m2m_model_path}")
    except Exception as e:
        logger.error(f"无法加载 mBART 模型: {e}")
        raise

    m2m_model.to(device)
    log_memory_usage()
    m2m_model.eval()
    return m2m_model, m2m_tokenizer

# 批量翻译函数（使用 mBART 模型）
def batch_translate(texts, m2m_model, m2m_tokenizer, device, src_lang="en", tgt_lang="zh"):
    logger.info(f"翻译输入（{src_lang} -> {tgt_lang}）: {texts[:2]}")
    inputs = m2m_tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    # mBART 使用 lang_code 进行语言指定
    src_lang_code = "en_XX" if src_lang == "en" else "zh_CN" if src_lang == "zh" else src_lang
    tgt_lang_code = "zh_CN" if tgt_lang == "zh" else "en_XX" if tgt_lang == "en" else tgt_lang
    translated = m2m_model.generate(
        **inputs,
        forced_bos_token_id=m2m_tokenizer.lang_code_to_id[tgt_lang_code],
        max_length=128
    )
    result = [m2m_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    result = [text if text.strip() else "翻译失败" for text in result]
    logger.info(f"翻译输出: {result[:2]}")
    return result

# 混合式回译函数（使用 mBART 模型）
translation_cache = {}
def enhanced_back_translate(texts, m2m_model, m2m_tokenizer, device, cultural_dict, batch_size=16):
    # 保护文化负载词
    processed_texts = []
    for text in texts:
        if text in translation_cache:
            processed_texts.append(text)
            continue
        for slang, info in cultural_dict["slang"].items():
            if slang in text:
                text = text.replace(slang, f"[{slang}]")
        for emoji, info in cultural_dict["emoji"].items():
            if emoji in text:
                text = text.replace(emoji, f"[{emoji}]")
        processed_texts.append(text)
    
    # 翻译到中文
    translated_texts = []
    for i in range(0, len(processed_texts), batch_size):
        batch_texts = processed_texts[i:i + batch_size]
        translated_batch = batch_translate(batch_texts, m2m_model, m2m_tokenizer, device, src_lang="en", tgt_lang="zh")
        translated_texts.extend(translated_batch)
        torch.cuda.empty_cache()
        gc.collect()
    
    # 回译到英文
    result_texts = []
    for i in range(0, len(translated_texts), batch_size):
        batch_texts = translated_texts[i:i + batch_size]
        back_translated_batch = batch_translate(batch_texts, m2m_model, m2m_tokenizer, device, src_lang="zh", tgt_lang="en")
        result_texts.extend(back_translated_batch)
        torch.cuda.empty_cache()
        gc.collect()
    
    # 恢复文化负载词并增强情感
    final_texts = []
    for result in result_texts:
        for slang, info in cultural_dict["slang"].items():
            if f"[{slang}]" in result:
                sentiment = info["sentiment"]
                result = result.replace(f"[{slang}]", f"{slang} ({'positive' if sentiment == 'positive' else 'negative' if sentiment == 'negative' else 'neutral'})")
        for emoji, info in cultural_dict["emoji"].items():
            if f"[{emoji}]" in result:
                sentiment = info["sentiment"]
                result = result.replace(f"[{emoji}]", f"{emoji} ({'positive' if sentiment == 'positive' else 'negative' if sentiment == 'negative' else 'neutral'})")
        final_texts.append(result)
    
    return final_texts

# 延迟加载 SentenceTransformer 模型
def load_sentence_transformer():
    logger.info("加载 SentenceTransformer 模型...")
    try:
        sentence_transformer = SentenceTransformer("/gemini/code/models/paraphrase-multilingual-mpnet-base-v2")
        logger.info("成功加载模型: /gemini/code/models/paraphrase-multilingual-mpnet-base-v2")
    except Exception as e:
        logger.warning(f"加载模型 /gemini/code/models/paraphrase-multilingual-mpnet-base-v2 失败: {e}")
        logger.info("回退到模型: /gemini/code/sentence_transformer_finetuned")
        sentence_transformer = SentenceTransformer("/gemini/code/sentence_transformer_finetuned")

    # 将 SentenceTransformer 模型加载到 GPU 上，加速嵌入提取
    sentence_transformer.to(device)
    log_memory_usage()
    return sentence_transformer

# 单模型预测函数
def predict_single_model(model, tokenizer, text, device):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            logger.info(f"Logits: {logits.cpu().numpy()}")
            probs = logits.softmax(dim=-1)  # [neg, pos]
            # 调整阈值以增加正类预测
            threshold = 0.5  # 从 0.6 调整到 0.5
            adjusted_probs = probs.clone()
            adjusted_probs[0, 1] = 1 if probs[0, 1] > threshold else 0
            adjusted_probs[0, 0] = 1 - adjusted_probs[0, 1]
        return adjusted_probs.cpu().numpy()[0]
    except Exception as e:
        logger.error(f"单模型预测失败: {e}")
        return np.array([0.5, 0.5])  # 返回默认概率

# 批量预测函数（加权投票）
def predict_batch(texts, models, tokenizers, weights, device, batch_size=32, use_enhanced_translation=False, m2m_model=None, m2m_tokenizer=None, cultural_dict=None, enhanced_texts=None):
    all_labels = {"xlm": [], "deepseek": [], "ensemble": []}
    all_probs = {"xlm": [], "deepseek": [], "ensemble": []}
    final_texts = []
    
    start_time = time.time()
    if use_enhanced_translation:
        if enhanced_texts is not None:
            texts = enhanced_texts
            logger.info("使用预翻译的文本进行推理")
        else:
            texts = enhanced_back_translate(texts, m2m_model, m2m_tokenizer, device, cultural_dict, batch_size=batch_size)
        final_texts.extend(texts)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{len(texts)//batch_size + 1}，样本范围: {i}-{min(i + batch_size, len(texts))}")
        
        batch_preds = {name: [] for name in models}
        for name in models:
            try:
                model_device = device  # 所有模型都在 GPU 上
                models[name].to(model_device)
                inputs = tokenizers[name](
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(model_device)
                with torch.no_grad():
                    outputs = models[name](**inputs).logits.softmax(dim=-1).cpu().numpy()
                batch_preds[name] = outputs
            except Exception as e:
                logger.error(f"模型 {name} 推理失败: {e}")
                batch_preds[name] = np.array([[0.5, 0.5]] * len(batch_texts))
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage()
        
        for j in range(len(batch_texts)):
            for name in models:
                pred = batch_preds[name][j]
                label = 1 if pred[1] > pred[0] else 0
                all_labels[name].append(label)
                all_probs[name].append(pred)
            final_pred = np.sum([weights[name] * batch_preds[name][j] for name in models], axis=0)
            label = 1 if final_pred[1] > final_pred[0] else 0
            all_labels["ensemble"].append(label)
            all_probs["ensemble"].append(final_pred)
    
    end_time = time.time()
    total_time = end_time - start_time
    batches = (len(texts) + batch_size - 1) // batch_size
    avg_time_per_batch = total_time / batches
    logger.info(f"推理 {len(texts)} 个样本耗时: {total_time:.2f} 秒，平均每批次耗时: {avg_time_per_batch:.2f} 秒")
    return all_labels, all_probs, final_texts if use_enhanced_translation else None

# Stacking 集成预测函数
def predict_stacking(texts, models, tokenizers, sentence_transformer, device, batch_size=32, use_enhanced_translation=False, m2m_model=None, m2m_tokenizer=None, cultural_dict=None, enhanced_texts=None):
    all_probs = {"xlm": [], "deepseek": []}
    
    if use_enhanced_translation:
        if enhanced_texts is not None:
            texts = enhanced_texts
            logger.info("使用预翻译的文本进行推理")
        else:
            texts = enhanced_back_translate(texts, m2m_model, m2m_tokenizer, device, cultural_dict, batch_size=batch_size)
    
    # 获取模型预测概率
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"处理批次 {i//batch_size + 1}/{len(texts)//batch_size + 1}，样本范围: {i}-{min(i + batch_size, len(texts))}")
        for name in models:
            try:
                model_device = device  # 所有模型都在 GPU 上
                models[name].to(model_device)
                inputs = tokenizers[name](
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(model_device)
                with torch.no_grad():
                    outputs = models[name](**inputs).logits.softmax(dim=-1).cpu().numpy()
                all_probs[name].extend(outputs)
            except Exception as e:
                logger.error(f"模型 {name} 推理失败: {e}")
                all_probs[name].extend([[0.5, 0.5]] * len(batch_texts))
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage()
    
    # 获取 SentenceTransformer 嵌入
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        logger.info(f"提取嵌入批次 {i//batch_size + 1}/{len(texts)//batch_size + 1}，样本范围: {i}-{min(i + batch_size, len(texts))}")
        try:
            batch_embeddings = sentence_transformer.encode(batch_texts, convert_to_numpy=True, device=device)
            embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"SentenceTransformer 嵌入提取失败: {e}")
            embeddings.extend([np.zeros(768)] * len(batch_texts))
    embeddings = np.array(embeddings)
    
    # 合并特征（模型概率 + 嵌入）
    X = np.hstack([np.array(all_probs["xlm"]), np.array(all_probs["deepseek"]), embeddings])
    meta_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=4, learning_rate=0.005, n_estimators=100)
    meta_model.fit(X[:len(test_labels)], test_labels)
    preds = meta_model.predict(X)
    return [1 if p == 1 else 0 for p in preds], X

# 主函数：运行三组实验
def main():
    global test_labels
    # 加载测试数据集
    logger.info("开始加载测试数据集...")
    test_texts, test_labels, enhanced_texts = load_data("/gemini/code/test_tweets.csv", max_samples=2000, is_train=False)
    logger.info(f"测试数据集加载完成。样本数量: {len(test_texts)}")

    # 加载模型和分词器
    models, tokenizers = load_models_and_tokenizers()
    cultural_dict = load_cultural_dict("/gemini/code/cultural_dict.json")
    
    # 初始权重（调整以提升 DeepSeek 贡献）
    weights = {"xlm": 0.4, "deepseek": 0.6}
    logger.info(f"初始权重: {weights}")

    # 第一组测试：加权投票，不启用翻译增强
    logger.info("\n第一组测试（加权投票，不启用翻译增强）：")
    labels, probs, _ = predict_batch(test_texts, models, tokenizers, weights, device, use_enhanced_translation=False)
    for model in ["xlm", "deepseek", "ensemble"]:
        pred_labels = labels[model]
        logger.info(f"\n{model.upper()} 结果:")
        logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
        logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
        logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
        logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

    # 第二组测试：加权投票，启用翻译增强
    logger.info("\n第二组测试（加权投票，启用翻译增强）：")
    m2m_model, m2m_tokenizer = load_m2m_model()
    labels_enhanced, probs_enhanced, enhanced_texts = predict_batch(test_texts, models, tokenizers, weights, device, use_enhanced_translation=True, m2m_model=m2m_model, m2m_tokenizer=m2m_tokenizer, cultural_dict=cultural_dict, enhanced_texts=enhanced_texts)
    for model in ["xlm", "deepseek", "ensemble"]:
        pred_labels = labels_enhanced[model]
        logger.info(f"\n{model.upper()} 结果:")
        logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
        logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
        logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
        logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

    # 第三组测试：Stacking 集成，启用翻译增强
    logger.info("\n第三组测试（Stacking 集成，启用翻译增强）：")
    sentence_transformer = load_sentence_transformer()
    start_time = time.time()
    labels_stacking, probs_stacking = predict_stacking(test_texts, models, tokenizers, sentence_transformer, device, use_enhanced_translation=True, m2m_model=m2m_model, m2m_tokenizer=m2m_tokenizer, cultural_dict=cultural_dict, enhanced_texts=enhanced_texts)
    end_time = time.time()
    total_time = end_time - start_time
    batches = (len(test_texts) + 64 - 1) // 64
    avg_time_per_batch = total_time / batches
    logger.info(f"推理 {len(test_texts)} 个样本耗时: {total_time:.2f} 秒，平均每批次耗时: {avg_time_per_batch:.2f} 秒")
    pred_labels = labels_stacking
    logger.info("\nSTACKING 结果:")
    logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
    logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
    logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
    logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

if __name__ == "__main__":
    main()