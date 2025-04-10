import pandas as pd
from sacrebleu import corpus_bleu
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from sentence_transformers import SentenceTransformer
import torch
import nltk
import os
from nltk.tokenize import word_tokenize
import emoji
import json
import logging
from tqdm import tqdm
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
nltk.data.path.append('/root/nltk_data')
for resource in ['punkt', 'punkt_tab']:
    if not os.path.exists(f'/root/nltk_data/tokenizers/{resource}'):
        raise FileNotFoundError(f"NLTK 资源 {resource} 未找到")

def protect_currency(text):
    currency_pattern = r'\$\d+(?:\.\d+)?[kKmM]?\b'
    matches = re.findall(currency_pattern, text)
    for i, match in enumerate(matches):
        text = text.replace(match, f"[CURRENCY_{i}]")
    return text, matches

def restore_currency(text, matches):
    if not isinstance(text, str): text = str(text)
    for i, match in enumerate(matches):
        text = text.replace(f"[CURRENCY_{i}]", match)
    return text

def protect_entities(text):
    try:
        import spacy
        nlp = spacy.load('/root/miniconda3/lib/python3.11/site-packages/spacy/data/en_core_web_lg')
        doc = nlp(text)
        entities = [ent.text for ent in doc.ents if ent.label_ in ["ORG", "PRODUCT", "PERSON", "GPE", "NORP"]]
        for i, entity in enumerate(entities):
            text = text.replace(entity, f"[ENTITY_{i}]")
        return text, entities
    except Exception:
        return text, []

def restore_entities(text, entities):
    if not isinstance(text, str): text = str(text)
    for i, entity in enumerate(entities):
        text = text.replace(f"[ENTITY_{i}]", entity)
    return text

def normalize_entity_tags(text):
    if not isinstance(text, str): text = str(text)
    text = re.sub(r'\[(?:Enity|entity|ENTITY|实体)_(\d+)\]', r'[ENTITY_\1]', text, flags=re.IGNORECASE)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

def clean_repeated_phrases(text):
    if not isinstance(text, str): text = str(text)
    words = text.split()
    if len(words) > 1:
        seen = set()
        result = []
        for w in words:
            if w not in seen or len(result) == 0 or result[-1] != w:
                seen.add(w)
                result.append(w)
        return ' '.join(result)
    return text

def preprocess_emoji(text):
    if not isinstance(text, str): text = str(text)
    emoji_dict = {'👍': ' positive ', '👎': ' negative ', '😊': ' happy ', '❤️': ' love ', '🌟': ' star '}
    for emoji_char, replacement in emoji_dict.items():
        text = text.replace(emoji_char, replacement)
    text = emoji.replace_emoji(text, replace='')
    return text

def preprocess_text(text):
    if not isinstance(text, str): text = str(text)
    text = preprocess_emoji(text)
    text, currency_matches = protect_currency(text)
    text, entities = protect_entities(text)
    return text, currency_matches, entities

def postprocess_text(text, currency_matches, entities):
    if not isinstance(text, str): text = str(text)
    text = normalize_entity_tags(text)
    text = restore_currency(text, currency_matches)
    text = clean_repeated_phrases(text)
    text = re.sub(r'(\d+)k\b', lambda m: f"{int(m.group(1)) * 1000}", text)
    text = re.sub(r'(\d+\.\d+)m\b', lambda m: f"{float(m.group(1)) * 1000000:.0f}", text)
    text = re.sub(r'(\d+)m\b', lambda m: f"{int(m.group(1)) * 1000000}", text)
    if text.lower() in ['yes, i am', 'yes']: text = 'true'
    return text

def translate_text(texts, model_path, tokenizer, src_lang, tgt_lang, batch_size=32):
    if not os.path.exists(model_path):
        logging.error(f"模型路径 {model_path} 不存在")
        return texts
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MBartForConditionalGeneration.from_pretrained(model_path).to(device)
    tokenizer.src_lang = src_lang
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"翻译 {src_lang} 到 {tgt_lang}"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
            num_beams=8,
            max_length=256,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translated_texts

def back_translation_analysis(df, sample_size=5):
    logging.info("开始翻译偏差分析...")
    sample_texts = df['cleaned_text'].head(sample_size).tolist()
    processed_texts = [preprocess_text(t) for t in sample_texts]
    cleaned_texts = [pt[0] for pt in processed_texts]
    
    model_path = "/gemini/code/models/mbart-large-50-many-to-many-mmt"
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)
    
    zh_texts = translate_text(cleaned_texts, model_path, tokenizer, "en_XX", "zh_CN")
    zh_texts = [normalize_entity_tags(t) for t in zh_texts]
    logging.info(f"中文翻译结果: {zh_texts}")
    
    back_translated_texts = translate_text(zh_texts, model_path, tokenizer, "zh_CN", "en_XX")
    back_translated_texts = [normalize_entity_tags(t) for t in back_translated_texts]
    logging.info(f"回译结果: {back_translated_texts}")
    
    zh_texts = [postprocess_text(zh, c, e) for zh, (_, c, e) in zip(zh_texts, processed_texts)]
    back_translated_texts = [postprocess_text(back, c, e) for back, (_, c, e) in zip(back_translated_texts, processed_texts)]
    
    back_translated_for_bleu = [normalize_entity_tags(restore_entities(t, pt[2])) for t, pt in zip(back_translated_texts, processed_texts)]
    bleu_score = corpus_bleu(back_translated_for_bleu, [sample_texts]).score
    logging.info("翻译偏差分析完成")
    return sample_texts, zh_texts, back_translated_texts, bleu_score, processed_texts

def compute_similarity(texts1, texts2, processed_texts, model_path="/gemini/code/models/paraphrase-multilingual-mpnet-base-v2"):
    logging.info("开始计算相似度...")
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if not os.path.exists(model_path):
            model_path = "/gemini/code/models/all-MiniLM-L6-v2"
        model = SentenceTransformer(model_path, device=device)
        texts1 = [normalize_entity_tags(restore_entities(t, e)) for t, e in zip(texts1, [pt[2] for pt in processed_texts])]
        texts2 = [normalize_entity_tags(restore_entities(t, e)) for t, e in zip(texts2, [pt[2] for pt in processed_texts])]
        embeddings1 = model.encode(texts1, show_progress_bar=True, convert_to_tensor=True)
        embeddings2 = model.encode(texts2, show_progress_bar=True, convert_to_tensor=True)
        similarities = torch.cosine_similarity(embeddings1, embeddings2)
        logging.info("相似度计算完成")
        return similarities.tolist()
    except Exception as e:
        logging.error(f"相似度计算失败: {e}")
        return [0.0] * len(texts1)

def extract_slang_and_emoji(texts):
    logging.info("开始文化建模...")
    slang_dict = {}
    emoji_dict = {}
    slang_list = {'lol', 'wtf', 'nah', 'y\'all', 'bruh', 'idk', 'cant', 'wont', 'cool', 'bro', 'dude', 'nope', 'lit'}
    for text in tqdm(texts, desc="处理文本"):
        tokens = word_tokenize(text.lower())
        for token in tokens:
            if token in slang_list:
                slang_dict[token] = slang_dict.get(token, 0) + 1
        emojis = emoji.distinct_emoji_list(text)
        for e in emojis:
            emoji_dict[e] = emoji_dict.get(e, 0) + 1
    logging.info("文化建模完成")
    return slang_dict, emoji_dict

if __name__ == "__main__":
    logging.info(f"GPU 可用: {torch.cuda.is_available()}")
    file_path = '/gemini/code/optimized_tweets_sentiment.csv'
    
    try:
        df = pd.read_csv(file_path)
        logging.info(f"数据量: {len(df)}")
        logging.info("数据预览:\n" + df.head().to_string())
    except Exception as e:
        logging.error(f"加载数据失败: {e}")
        raise

    logging.info("\n=== 翻译偏差处理 ===")
    original, zh_texts, back_translated, bleu, processed_texts = back_translation_analysis(df, sample_size=5)
    print("原文:", original)
    print("中文翻译:", zh_texts)
    print("回译文本:", back_translated)
    print(f"BLEU 分数: {bleu}")

    logging.info("\n=== 跨语言对齐 ===")
    similarities = compute_similarity(original, zh_texts, processed_texts)
    print("文本对:", list(zip(original, zh_texts)))
    print(f"相似度: {similarities}")

    logging.info("\n=== 文化建模 ===")
    slang_dict, emoji_dict = extract_slang_and_emoji(df['cleaned_text'].tolist())
    print(f"俚语词典 (前5): {dict(list(slang_dict.items())[:5])}")
    print(f"表情符号词典 (前5): {dict(list(emoji_dict.items())[:5]) if emoji_dict else '无表情符号'}")
    with open('cultural_dict.json', 'w', encoding='utf-8') as f:
        json.dump({'slang': slang_dict, 'emoji': emoji_dict}, f, ensure_ascii=False)
    logging.info("文化词典已保存至 cultural_dict.json")

    logging.info("\n保存增强数据...")
    df['zh_translated_text'] = pd.Series(zh_texts + [''] * (len(df) - len(zh_texts)))
    df['back_translated_text'] = pd.Series(back_translated + [''] * (len(df) - len(back_translated)))
    df.to_csv('enhanced_tweets.csv', index=False)
    logging.info("增强数据已保存至 enhanced_tweets.csv")