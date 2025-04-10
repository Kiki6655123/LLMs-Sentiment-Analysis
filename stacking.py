import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, MBartForConditionalGeneration, MBart50TokenizerFast
from peft import PeftModel  # 引入 peft 库以加载 LoRA 模型
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import time
import logging
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/gemini/code/stacking_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"使用设备: {device}")

# 1. 加载微调模型和分词器
logger.info("开始加载模型和分词器...")
# 加载 XLM-RoBERTa
xlm_model = AutoModelForSequenceClassification.from_pretrained("/gemini/code/finetuned_xlm_roberta")
xlm_tokenizer = AutoTokenizer.from_pretrained("/gemini/code/finetuned_xlm_roberta")

# 加载 DeepSeek（基础模型 + LoRA 适配器）
base_model = AutoModelForSequenceClassification.from_pretrained(
    "/gemini/code/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True,
    num_labels=2  # 确保二分类
)
deepseek_model = PeftModel.from_pretrained(
    base_model,
    "/gemini/code/instruction_finetuned_deepseek",
    trust_remote_code=True
)
deepseek_tokenizer = AutoTokenizer.from_pretrained(
    "/gemini/code/DeepSeek-R1-Distill-Qwen-7B",
    trust_remote_code=True
)

# 强制设置 DeepSeek 分词器的 pad_token 和 pad_token_id
if deepseek_tokenizer.pad_token is None:
    deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token  # 使用 eos_token 作为 pad_token
    deepseek_tokenizer.pad_token_id = deepseek_tokenizer.eos_token_id
    logger.info("为 DeepSeek 分词器设置 pad_token: %s", deepseek_tokenizer.pad_token)

# 如果仍未生效，手动添加 pad_token
if deepseek_tokenizer.pad_token is None:
    deepseek_tokenizer.add_special_tokens({'pad_token': '<pad>'})
    deepseek_tokenizer.pad_token_id = deepseek_tokenizer.convert_tokens_to_ids('<pad>')
    deepseek_model.resize_token_embeddings(len(deepseek_tokenizer))
    logger.info("手动为 DeepSeek 分词器添加 pad_token: <pad>")

# 确保模型配置中包含 pad_token_id
if not hasattr(deepseek_model.config, 'pad_token_id') or deepseek_model.config.pad_token_id is None:
    deepseek_model.config.pad_token_id = deepseek_tokenizer.pad_token_id
    logger.info("为 DeepSeek 模型设置 pad_token_id: %d", deepseek_model.config.pad_token_id)

models = {
    "xlm": xlm_model,
    "deepseek": deepseek_model
}
tokenizers = {
    "xlm": xlm_tokenizer,
    "deepseek": deepseek_tokenizer
}

# 加载 mBART_finetuned_v5
logger.info("加载 mBART 模型...")
mbart_model_path = "/gemini/code/models/mbart_finetuned_v5"
if not os.path.exists(mbart_model_path):
    logger.error(f"mBART 模型路径 {mbart_model_path} 不存在！")
    raise FileNotFoundError(f"mBART 模型路径 {mbart_model_path} 不存在！")
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_path)
mbart_tokenizer = MBart50TokenizerFast.from_pretrained(mbart_model_path)
logger.info("mBART 模型加载完成。")

# 加载 SentenceTransformer
sentence_transformer = SentenceTransformer("/gemini/code/sentence_transformer_finetuned")

# 将模型移到设备并设置为评估模式
for name, model in models.items():
    model.to(device)
    model.eval()
mbart_model.to(device)
mbart_model.eval()
sentence_transformer.to(device)
logger.info("模型和分词器加载完成。")

# 2. 初始权重（调整权重以提升性能）
weights = {
    "xlm": 0.8,      # XLM-RoBERTa 权重提高
    "deepseek": 0.2   # DeepSeek 权重降低
}
logger.info(f"初始权重: {weights}")

# 3. 文化词典（简化版）
cultural_dict = {
    "slang": {
        "cant": {"freq": 383, "sentiment": "negative"},
        "bro": {"freq": 111, "sentiment": "neutral"},
        "wont": {"freq": 146, "sentiment": "negative"},
        "idk": {"freq": 42, "sentiment": "neutral"},
        "bruh": {"freq": 17, "sentiment": "negative"},
        "nah": {"freq": 43, "sentiment": "negative"},
        "lol": {"freq": 156, "sentiment": "positive"},
        "lit": {"freq": 4, "sentiment": "positive"},
        "nope": {"freq": 15, "sentiment": "negative"},
        "dude": {"freq": 53, "sentiment": "neutral"},
        "cool": {"freq": 47, "sentiment": "positive"},
        "wtf": {"freq": 35, "sentiment": "negative"}
    },
    "emoji": {
        "😃": {"freq": 8, "sentiment": "positive"},
        "😎": {"freq": 46, "sentiment": "positive"},
        "🎮": {"freq": 9, "sentiment": "neutral"},
        "🔥": {"freq": 575, "sentiment": "positive"},
        "👍": {"freq": 53, "sentiment": "positive"},
        "🔽": {"freq": 4, "sentiment": "negative"},
        "🌱": {"freq": 27, "sentiment": "positive"},
        "💼": {"freq": 104, "sentiment": "neutral"},
        "🙌": {"freq": 20, "sentiment": "positive"},
        "💡": {"freq": 36, "sentiment": "positive"},
        "😂": {"freq": 293, "sentiment": "positive"},
        "😭": {"freq": 317, "sentiment": "negative"},
        "💔": {"freq": 191, "sentiment": "negative"},
        "🙏": {"freq": 128, "sentiment": "neutral"},
        "😡": {"freq": 66, "sentiment": "negative"},
        "💪": {"freq": 187, "sentiment": "positive"},
        "😊": {"freq": 54, "sentiment": "positive"}
    }
}
logger.info("文化词典加载完成。")

# 4. 单模型预测函数
def predict_single_model(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs).logits.softmax(dim=-1)  # [neg, pos]
    return outputs.cpu().numpy()[0]

# 5. 批量翻译函数（移除 Prompt）
def batch_translate(texts, mbart_model, mbart_tokenizer, device, src_lang="en_XX", tgt_lang="zh_CN"):
    mbart_tokenizer.src_lang = src_lang
    inputs = mbart_tokenizer(texts, return_tensors="pt", truncation=True, padding=True).to(device)
    translated = mbart_model.generate(**inputs, forced_bos_token_id=mbart_tokenizer.lang_code_to_id[tgt_lang], max_length=128)
    result = [mbart_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
    return result

# 6. 混合式回译函数（移除中途回译结果）
translation_cache = {}
def enhanced_back_translate(texts, mbart_model, mbart_tokenizer, device, cultural_dict, batch_size=16):
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
        translated_batch = batch_translate(batch_texts, mbart_model, mbart_tokenizer, device, src_lang="en_XX", tgt_lang="zh_CN")
        translated_texts.extend(translated_batch)
        torch.cuda.empty_cache()  # 释放 GPU 内存
    
    # 回译到英文
    result_texts = []
    for i in range(0, len(translated_texts), batch_size):
        batch_texts = translated_texts[i:i + batch_size]
        back_translated_batch = batch_translate(batch_texts, mbart_model, mbart_tokenizer, device, src_lang="zh_CN", tgt_lang="en_XX")
        result_texts.extend(back_translated_batch)
        torch.cuda.empty_cache()  # 释放 GPU 内存
    
    # 恢复文化负载词并增强情感
    final_texts = []
    for result in result_texts:
        for slang, info in cultural_dict["slang"].items():
            if f"[{slang}]" in result:
                sentiment = info["sentiment"]
                result = result.replace(f"[{slang}]", f"{slang} ({'积极' if sentiment == 'positive' else '消极' if sentiment == 'negative' else '中性'})")
        for emoji, info in cultural_dict["emoji"].items():
            if f"[{emoji}]" in result:
                sentiment = info["sentiment"]
                result = result.replace(f"[{emoji}]", f"{emoji} ({'积极' if sentiment == 'positive' else '消极' if sentiment == 'negative' else '中性'})")
        final_texts.append(result)
    
    return final_texts

# 7. 获取 SentenceTransformer 嵌入
def get_sentence_embeddings(texts, sentence_transformer, batch_size=32):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = sentence_transformer.encode(batch_texts, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

# 8. 加权投票预测函数
def predict_ensemble(text, models, tokenizers, weights, device, use_enhanced_translation=False):
    if use_enhanced_translation:
        text = enhanced_back_translate([text], mbart_model, mbart_tokenizer, device, cultural_dict)[0]
    
    preds = {}
    for name in models:
        preds[name] = predict_single_model(models[name], tokenizers[name], text, device)
    
    final_pred = np.sum([weights[name] * preds[name] for name in models], axis=0)
    label = "积极" if final_pred[1] > final_pred[0] else "消极"
    return label, final_pred, text

# 9. Stacking 集成预测函数（结合 SentenceTransformer 嵌入）
def predict_stacking(texts, models, tokenizers, sentence_transformer, device, batch_size=256, use_enhanced_translation=False):
    all_probs = {"xlm": [], "deepseek": []}
    
    if use_enhanced_translation:
        texts = enhanced_back_translate(texts, mbart_model, mbart_tokenizer, device, cultural_dict, batch_size=16)
    
    # 获取模型预测概率
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        for name in models:
            inputs = tokenizers[name](
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = models[name](**inputs).logits.softmax(dim=-1).cpu().numpy()
            all_probs[name].extend(outputs)
        torch.cuda.empty_cache()  # 释放 GPU 内存
    
    # 获取 SentenceTransformer 嵌入
    embeddings = get_sentence_embeddings(texts, sentence_transformer, batch_size=32)
    
    # 合并特征（模型概率 + 嵌入）
    X = np.hstack([np.array(all_probs["xlm"]), np.array(all_probs["deepseek"]), embeddings])
    meta_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', max_depth=12, learning_rate=0.02, n_estimators=500)
    meta_model.fit(X[:len(test_labels)], test_labels)  # 使用测试集标签训练元模型
    preds = meta_model.predict(X)
    return ["积极" if p == 1 else "消极" for p in preds], X

# 10. 批量预测函数（加权投票）
def predict_batch(texts, models, tokenizers, weights, device, batch_size=256, use_enhanced_translation=False):
    all_labels = {"xlm": [], "deepseek": [], "ensemble": []}
    all_probs = {"xlm": [], "deepseek": [], "ensemble": []}
    enhanced_texts = []
    
    start_time = time.time()
    if use_enhanced_translation:
        texts = enhanced_back_translate(texts, mbart_model, mbart_tokenizer, device, cultural_dict, batch_size=16)
        enhanced_texts.extend(texts)
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        batch_preds = {name: [] for name in models}
        for name in models:
            inputs = tokenizers[name](
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(device)
            with torch.no_grad():
                outputs = models[name](**inputs).logits.softmax(dim=-1).cpu().numpy()
            batch_preds[name] = outputs
        torch.cuda.empty_cache()  # 释放 GPU 内存
        
        for j in range(len(batch_texts)):
            for name in models:
                pred = batch_preds[name][j]
                label = "积极" if pred[1] > pred[0] else "消极"
                all_labels[name].append(label)
                all_probs[name].append(pred)
            final_pred = np.sum([weights[name] * batch_preds[name][j] for name in models], axis=0)
            label = "积极" if final_pred[1] > final_pred[0] else "消极"
            all_labels["ensemble"].append(label)
            all_probs["ensemble"].append(final_pred)
    
    end_time = time.time()
    logger.info(f"推理 {len(texts)} 个样本耗时: {end_time - start_time:.2f} 秒")
    return all_labels, all_probs, enhanced_texts if use_enhanced_translation else None

# 11. 主函数：加载数据、预测与对比
def main():
    global test_labels  # 全局变量，用于 Stacking
    # 加载测试数据集并确保二分类
    logger.info("开始加载测试数据集...")
    data = pd.read_csv("/gemini/code/new_tweets_v2.csv")
    data = data[data["label"] != 2].reset_index(drop=True)
    data = data.dropna(subset=['cleaned_text'])  # 清洗 NaN 值
    logger.info("测试数据集加载完成（二分类）。样本预览:")
    logger.info(f"\n{data.head().to_string()}")
    logger.info("标签分布:")
    logger.info(f"\n{data['label'].value_counts().to_string()}")

    # 测试集评估
    test_texts = data["cleaned_text"].tolist()
    test_labels = data["label"].values

    # 单条预测示例
    sample_text = "blockchain is lit 🔥 and bro this is cool 👍"
    label, probs, enhanced = predict_ensemble(sample_text, models, tokenizers, weights, device, use_enhanced_translation=True)
    logger.info("\n单条预测（使用增强翻译）:")
    logger.info(f"原始文本: {sample_text}")
    logger.info(f"增强文本: {enhanced}")
    logger.info(f"预测结果: {label}, 概率: {probs}")

    # 不使用增强翻译（加权投票）
    logger.info("\n测试集性能（不使用增强翻译，加权投票）:")
    labels, probs, _ = predict_batch(test_texts, models, tokenizers, weights, device, use_enhanced_translation=False)
    for model in ["xlm", "deepseek", "ensemble"]:
        pred_labels = [1 if p == "积极" else 0 for p in labels[model]]
        logger.info(f"\n{model.upper()} 结果:")
        logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
        logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
        logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
        logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

    # 使用增强翻译（加权投票）
    logger.info("\n测试集性能（使用增强翻译，加权投票）:")
    labels_enhanced, probs_enhanced, enhanced_texts = predict_batch(test_texts, models, tokenizers, weights, device, use_enhanced_translation=True)
    for model in ["xlm", "deepseek", "ensemble"]:
        pred_labels = [1 if p == "积极" else 0 for p in labels_enhanced[model]]
        logger.info(f"\n{model.upper()} 结果:")
        logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
        logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
        logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
        logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

    # 使用增强翻译（Stacking 集成）
    logger.info("\n测试集性能（使用增强翻译，Stacking 集成）:")
    labels_stacking, probs_stacking = predict_stacking(test_texts, models, tokenizers, sentence_transformer, device, use_enhanced_translation=True)
    pred_labels = [1 if p == "积极" else 0 for p in labels_stacking]
    logger.info("\nSTACKING 结果:")
    logger.info(f"准确率: {accuracy_score(test_labels, pred_labels):.4f}")
    logger.info(f"F1 分数: {f1_score(test_labels, pred_labels):.4f}")
    logger.info(f"精确率: {precision_score(test_labels, pred_labels):.4f}")
    logger.info(f"召回率: {recall_score(test_labels, pred_labels):.4f}")

if __name__ == "__main__":
    main()