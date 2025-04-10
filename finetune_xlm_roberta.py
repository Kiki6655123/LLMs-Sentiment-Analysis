# finetune_xlm_roberta.py
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import logging

# 设置日志
logging.basicConfig(filename='/gemini/code/finetune_xlm.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("开始微调 XLM-RoBERTa")

# 加载抽样数据
try:
    df = pd.read_csv("/gemini/code/sampled_weibo_comments_10000.csv")
except FileNotFoundError as e:
    logging.error(f"数据文件未找到: {e}")
    raise

# 检查数据分布
label_dist = df['label'].value_counts()
logging.info(f"训练数据分布: {label_dist}")

train_df = df.sample(frac=0.8, random_state=42)
val_df = df.drop(train_df.index)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# 加载 tokenizer 和模型
model_name = "/gemini/code/xlm-roberta-base"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
except Exception as e:
    logging.error(f"模型加载失败: {e}")
    raise

# 预处理数据
def tokenize_function(examples):
    return tokenizer(examples["cleaned_review"], truncation=True, padding="max_length", max_length=16)  # 最小化序列长度

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 将 label 转换为 0 和 1
train_dataset = train_dataset.map(lambda x: {"labels": int(x["label"])})
val_dataset = val_dataset.map(lambda x: {"labels": int(x["label"])})
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/gemini/code/finetuned_xlm_roberta",
    overwrite_output_dir=True,
    num_train_epochs=15,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,  # 增加积累步数
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    learning_rate=3e-5,
    evaluation_strategy="steps",
    eval_steps=500,
    logging_dir='/gemini/code/finetune_xlm_logs',
    fp16=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# 创建 Trainer 并开始微调
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

logging.info("开始微调...")
try:
    trainer.train()
except Exception as e:
    logging.error(f"微调过程失败: {e}")
    raise

# 保存模型
model.save_pretrained("/gemini/code/finetuned_xlm_roberta")
tokenizer.save_pretrained("/gemini/code/finetuned_xlm_roberta")
logging.info("微调完成，模型保存至 /gemini/code/finetuned_xlm_roberta")