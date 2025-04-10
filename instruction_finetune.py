import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, BitsAndBytesConfig
from datasets import Dataset
import torch
import torch.nn as nn
import logging
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from peft import LoraConfig, get_peft_model, PeftModel

# 设置日志
log_file = f'/gemini/code/instruction_finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("继续微调：从检查点恢复并继续训练")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")

# 定义全局 bnb_config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# 加载 SentenceTransformer 模型（第一层微调结果）
sentence_model = SentenceTransformer('/gemini/code/sentence_transformer_finetuned', device=device)

# 加载社交媒体对话数据
def load_social_media_data(file_path):
    df = pd.read_csv(file_path)
    df = df[df["label"].isin([0, 1])]
    label_counts = df["label"].value_counts().to_dict()
    logging.info(f"{file_path} label 分布（过滤后）: {label_counts}")
    logging.info(f"数据集列: {df.columns.tolist()}")
    if df.empty:
        raise ValueError(f"数据集 {file_path} 为空")
    if not all(col in df.columns for col in ["cleaned_review", "zh_translated_text", "label"]):
        raise ValueError(f"数据集缺少必要列，当前列: {df.columns.tolist()}")
    # 填充缺失值
    df["zh_translated_text"] = df["zh_translated_text"].fillna(df["cleaned_review"])
    return df[["cleaned_review", "zh_translated_text", "label"]]

# 构造 Prompt 数据（使用中文翻译）
def create_prompt_data(df):
    prompts = []
    for _, row in df.iterrows():
        text = row["zh_translated_text"] if pd.notnull(row["zh_translated_text"]) else row["cleaned_review"]
        label = row["label"]
        prompt = f"判断‘{text}’的情感并解释原因：积极（positive）或消极（negative）"
        label_text = "positive" if label == 1 else "negative"
        prompts.append({
            "prompt": prompt,
            "completion": f"情感：{label_text}，原因：{text} 表达了{'积极' if label == 1 else '消极'}的情绪。",
            "text": text
        })
    dataset = Dataset.from_pandas(pd.DataFrame(prompts))
    logging.info(f"构造后的数据集列: {dataset.column_names}")
    if not dataset:
        raise ValueError("构造后的数据集为空")
    return dataset

# 数据预处理
def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["prompt"],
        truncation=True,
        padding="max_length",
        max_length=256,
        pad_to_max_length=True,
        return_tensors="pt"
    )

# 加载模型和 tokenizer（4-bit 量化 + LoRA）
def load_model_and_tokenizer(base_path, checkpoint_path=None):
    try:
        logging.info("开始加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info(f"未定义填充标记，使用 EOS 标记: {tokenizer.pad_token}")
        logging.info(f"Tokenizer 配置: pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")
        
        logging.info("开始加载模型（4-bit 量化）...")
        model = AutoModelForSequenceClassification.from_pretrained(
            base_path,
            num_labels=2,
            device_map="auto",
            quantization_config=bnb_config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        logging.info(f"模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # 如果提供了检查点路径，则从检查点恢复 LoRA 适配器
        if checkpoint_path:
            logging.info(f"从检查点 {checkpoint_path} 恢复 LoRA 适配器...")
            model = PeftModel.from_pretrained(
                model,
                checkpoint_path,
                device_map="auto",
                trust_remote_code=True
            )
            logging.info("LoRA 适配器恢复完成")
        else:
            # 添加 LoRA 适配器
            logging.info("添加 LoRA 适配器...")
            lora_config = LoraConfig(
                r=64,  # 增加 r
                lora_alpha=32,  # 调整 lora_alpha
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, lora_config)
            # 确保 LoRA 参数可训练
            for name, param in model.named_parameters():
                if "lora" in name:
                    param.requires_grad = True
            logging.info("LoRA 适配器添加完成")
        
        return model, tokenizer
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise

# 初始化嵌入层（添加维度映射和批量处理）
def initialize_embeddings_with_sentence_transformer(model, tokenizer, sentence_model, dataset, top_k=10000):
    model.eval()
    with torch.no_grad():
        texts = dataset["text"]
        embeddings = sentence_model.encode(texts, convert_to_tensor=True, device=device, batch_size=128)
        embedding_layer = model.get_input_embeddings()
        embedding_dim = embedding_layer.embedding_dim  # 3584 for DeepSeek-R1-Distill-Qwen-7B
        sentence_embedding_dim = embeddings.shape[1]  # 768 for paraphrase-multilingual-mpnet-base-v2
        
        # 统计词频
        word_counts = {}
        for text in texts:
            tokens = tokenizer.tokenize(text)
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1
        
        # 选择高频词汇
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        high_freq_words = [word for word, _ in sorted_words]
        logging.info(f"选择 {len(high_freq_words)} 个高频词汇进行嵌入初始化")
        
        # 添加线性变换层，将 768 维映射到 3584 维
        linear_transform = nn.Linear(sentence_embedding_dim, embedding_dim).to(device)
        vocab = tokenizer.get_vocab()
        new_embeddings = embedding_layer.weight.data.clone()
        
        # 批量处理高频词汇
        batch_size = 128
        total_batches = (len(high_freq_words) + batch_size - 1) // batch_size
        for i in range(0, len(high_freq_words), batch_size):
            batch_words = high_freq_words[i:i + batch_size]
            logging.info(f"处理第 {i // batch_size + 1}/{total_batches} 批词汇")
            word_embeddings = sentence_model.encode(batch_words, convert_to_tensor=True, device=device, batch_size=batch_size)
            transformed_embeddings = linear_transform(word_embeddings)
            for word, embedding in zip(batch_words, transformed_embeddings):
                if word in vocab:
                    idx = vocab[word]
                    new_embeddings[idx] = embedding
        
        embedding_layer.weight.data = new_embeddings
    return model

# 计算困惑度
def compute_perplexity(model, dataset, tokenizer, device, max_samples=1000):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_size = 1  # 减小批量大小
    dataset = dataset.select(range(min(max_samples, len(dataset))))  # 仅使用前 max_samples 条
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    with torch.no_grad():
        for i in range(0, len(dataset), batch_size):
            batch_idx = i // batch_size + 1
            logging.info(f"计算困惑度：处理第 {batch_idx}/{total_batches} 批")
            batch = dataset[i:i+batch_size]
            inputs = {
                "input_ids": batch["input_ids"].clone().detach().to(device),
                "attention_mask": batch["attention_mask"].clone().detach().to(device),
                "labels": batch["labels"].clone().detach().to(device)
            }
            outputs = model(**inputs)
            loss = outputs.loss
            total_loss += loss.item() * inputs["input_ids"].size(0)
            total_tokens += inputs["attention_mask"].sum().item()
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity

# 加载数据
social_media_data = load_social_media_data("/gemini/code/train_enhanced_tweets.csv")
prompt_dataset = create_prompt_data(social_media_data)

# 加载模型（从检查点恢复）
base_model_path = "/gemini/code/DeepSeek-R1-Distill-Qwen-7B"
checkpoint_path = "/gemini/code/instruction_finetuned_deepseek/checkpoint-21930"
model, tokenizer = load_model_and_tokenizer(base_model_path, checkpoint_path)

# 初始化嵌入层（第一层结果迁移）
model = initialize_embeddings_with_sentence_transformer(model, tokenizer, sentence_model, prompt_dataset)
logging.info("嵌入层已使用 SentenceTransformer 初始化")
del sentence_model  # 释放 SentenceTransformer 模型
torch.cuda.empty_cache()  # 清理显存

# 预处理数据
logging.info("开始预处理数据...")
prompt_dataset = prompt_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
prompt_dataset = prompt_dataset.map(lambda x: {"labels": 1 if "positive" in x["completion"] else 0})
prompt_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
logging.info("数据预处理完成")

# 计算基础模型困惑度
base_perplexity = compute_perplexity(model, prompt_dataset, tokenizer, device)
logging.info(f"基础模型困惑度: {base_perplexity}")

# 设置微调参数
training_args = TrainingArguments(
    output_dir="/gemini/code/instruction_finetuned_deepseek",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=10,  # 增加到 10 个 epoch
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_dir='/gemini/code/instruction_finetune_logs',
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=False,
    gradient_checkpointing=True,
    optim="adamw_8bit",
    learning_rate=1e-5,  # 降低学习率
    resume_from_checkpoint=checkpoint_path  # 从检查点恢复
)

# 创建 Trainer
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if not prediction_loss_only:
            loss, logits, labels = outputs
            logging.debug(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
            logging.debug(f"logits sample: {logits[:5]}, labels sample: {labels[:5]}")
        return outputs

    def _load_optimizer_and_scheduler(self, checkpoint):
        # 跳过优化器状态加载，仅加载模型权重
        logging.info("跳过优化器状态加载，从头开始优化...")
        super()._load_optimizer_and_scheduler(None)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")  # 保持 Float 类型
        outputs = model(**inputs)
        logits = outputs.get("logits").to(torch.float32)  # 转换为 Float 类型
        # 计算类权重
        class_weights = torch.tensor([1.0, 1.27], device=logits.device, dtype=torch.float32)  # 保持 Float 类型
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=prompt_dataset,
    eval_dataset=prompt_dataset,
    compute_metrics=compute_metrics
)

# 微调模型
logging.info("开始继续微调...")
trainer.train(resume_from_checkpoint=checkpoint_path)
trainer.save_model("/gemini/code/instruction_finetuned_deepseek")
logging.info("继续微调完成，模型已保存至 /gemini/code/instruction_finetuned_deepseek")

# 计算微调模型困惑度
finetuned_model = model  # 直接使用 model，避免重复加载
finetuned_perplexity = compute_perplexity(finetuned_model, prompt_dataset, tokenizer, device)
logging.info(f"微调模型困惑度: {finetuned_perplexity}")

# 验证困惑度降低
perplexity_reduction = (base_perplexity - finetuned_perplexity) / base_perplexity * 100
logging.info(f"困惑度降低百分比: {perplexity_reduction:.2f}%")
if perplexity_reduction >= 17.2:
    logging.info("困惑度降低目标（17.2%）已达成")
else:
    logging.warning(f"困惑度降低未达目标（17.2%），实际降低: {perplexity_reduction:.2f}%")

# 评估 F1 值
def evaluate_and_plot(trainer, dataset, name):
    eval_results = trainer.evaluate()
    logging.info(f"{name} 评估结果: {eval_results}")
    print(f"{name} 评估结果:", eval_results)
    
    pred_output = trainer.predict(dataset)
    cm = confusion_matrix(dataset["labels"], pred_output.predictions.argmax(-1))
    plt.matshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.savefig(f"/gemini/code/{name}_cm.png")
    plt.close()
    
    predictions = torch.tensor(pred_output.predictions, dtype=torch.float32)
    probs = torch.softmax(predictions, dim=-1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(dataset["labels"], probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.legend()
    plt.title(f"ROC Curve - {name}")
    plt.savefig(f"/gemini/code/{name}_roc.png")
    plt.close()
    return eval_results

test_dataset = load_social_media_data("/gemini/code/test_enhanced_tweets.csv")
test_dataset = create_prompt_data(test_dataset)
test_dataset = test_dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)
test_dataset = test_dataset.map(lambda x: {"labels": 1 if "positive" in x["completion"] else 0})
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels", "prompt"])

eval_args = TrainingArguments(
    output_dir="/gemini/code/eval_deepseek_qwen_7b",
    per_device_eval_batch_size=1,
    logging_dir='/gemini/code/eval_deepseek_qwen_7b_logs',
    report_to="none",
    fp16=False,
    do_train=False,
    do_eval=True
)

base_trainer = CustomTrainer(
    model=model,
    args=eval_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
base_results = evaluate_and_plot(base_trainer, test_dataset, "base")

finetuned_trainer = CustomTrainer(
    model=finetuned_model,
    args=eval_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)
finetuned_results = evaluate_and_plot(finetuned_trainer, test_dataset, "finetuned")

base_f1 = base_results["eval_f1"]
finetuned_f1 = finetuned_results["eval_f1"]
logging.info(f"基础模型 F1 值: {base_f1 * 100:.1f}%")
logging.info(f"微调模型 F1 值: {finetuned_f1 * 100:.1f}%")
if base_f1 * 100 >= 78.5 and finetuned_f1 * 100 >= 86.9:
    logging.info("F1 值目标（78.5% -> 86.9%）已达成")
else:
    logging.warning(f"F1 值未达目标，实际值: {base_f1 * 100:.1f}% -> {finetuned_f1 * 100:.1f}%")