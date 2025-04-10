import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer, MT5ForConditionalGeneration, T5Tokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import Dataset
import torch
import logging
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 翻译函数（使用 M2M100 生成伪标签）
def translate_with_m2m100(texts, model, tokenizer, src_lang="en", tgt_lang="zh", batch_size=4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    tokenizer.src_lang = src_lang
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"翻译 {src_lang} 到 {tgt_lang} (M2M100)"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            num_beams=5,
            max_length=512,
            min_length=20,
            length_penalty=1.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
        torch.cuda.empty_cache()
    return translated_texts

# 数据预处理函数（用于 mT5 微调）
def preprocess_function(examples):
    inputs = [f"translate en to zh: {ex}" for ex in examples["source"]]
    targets = examples["target"]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 主函数
if __name__ == "__main__":
    logging.info(f"GPU 可用: {torch.cuda.is_available()}")

    # 清理 GPU 内存
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        used_memory = torch.cuda.memory_allocated(0) / 1024**3
        logging.info(f"初始显存: 总计 {total_memory:.2f} GiB, 已用 {used_memory:.2f} GiB")

    # 加载 Twitter 数据
    file_path = '/gemini/code/filtered_tweets_sentiment.csv'
    df = pd.read_csv(file_path)
    logging.info(f"原始数据量: {len(df)}")
    sample_size = 10000
    df_sample = df['cleaned_review'].head(sample_size).tolist()
    logging.info(f"采样数据量: {len(df_sample)}")

    # 加载 M2M100_418M 并生成伪标签
    m2m_model = M2M100ForConditionalGeneration.from_pretrained("/gemini/code/models/m2m100_418M")
    m2m_tokenizer = M2M100Tokenizer.from_pretrained("/gemini/code/models/m2m100_418M")
    pseudo_zh = translate_with_m2m100(df_sample, m2m_model, m2m_tokenizer, "en", "zh")
    train_df = pd.DataFrame({"source": df_sample, "target": pseudo_zh})
    train_df.to_csv("/gemini/code/pseudo_tweets_m2m100.tsv", sep="\t", index=False)
    logging.info("伪标签数据已保存至 /gemini/code/pseudo_tweets_m2m100.tsv")

    # 释放 M2M100 内存
    del m2m_model, m2m_tokenizer
    torch.cuda.empty_cache()

    # 加载微调数据
    train_df = pd.read_csv("/gemini/code/pseudo_tweets_m2m100.tsv", sep="\t")
    dataset = Dataset.from_pandas(train_df)
    logging.info(f"微调数据量: {len(dataset)}")

    # 加载 mT5-base 模型和分词器
    model = MT5ForConditionalGeneration.from_pretrained("/gemini/code/models/google-mt5-base")
    tokenizer = T5Tokenizer.from_pretrained("/gemini/code/models/google-mt5-base")
    logging.info("mT5-base 模型和分词器加载完成")

    # 数据预处理
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    train_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(0.8 * len(tokenized_dataset))))
    eval_dataset = tokenized_dataset.shuffle(seed=42).select(range(int(0.8 * len(tokenized_dataset)), len(tokenized_dataset)))
    logging.info(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(eval_dataset)}")

    # 设置训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="/gemini/code/models/mt5_finetuned_m2m100",
        eval_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_steps=500,
        save_total_limit=2,
        predict_with_generate=True,
        logging_dir="/gemini/code/logs",
        logging_steps=100,
        fp16=True if torch.cuda.is_available() else False,
    )

    # 初始化训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer
    )

    # 开始训练
    logging.info("开始微调 mT5-base...")
    trainer.train()

    # 保存模型
    model.save_pretrained("/gemini/code/models/mt5_finetuned_m2m100")
    tokenizer.save_pretrained("/gemini/code/models/mt5_finetuned_m2m100")
    logging.info("模型已保存至 /gemini/code/models/mt5_finetuned_m2m100")