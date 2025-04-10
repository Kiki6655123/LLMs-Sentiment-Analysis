# finetune_deepseek_qwen_7b.py
import pandas as pd
from transformers import AutoTokenizer, Qwen2ForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import logging
import os
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 设置日志
logging.basicConfig(filename='/gemini/code/finetune_deepseek_qwen_7b.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("开始微调 DeepSeek-R1-Distill-Qwen-7B with LoRA")

# 加载抽样数据
try:
    df = pd.read_csv("/gemini/code/sampled_weibo_comments_4000.csv")
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

# 加载 DeepSeek-R1-Distill-Qwen-7B tokenizer 和模型
model_path = "/gemini/code/DeepSeek-R1-Distill-Qwen-7B"
try:
    # 启用 4-bit 量化
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # 显式设置填充 token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS token 作为填充 token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = Qwen2ForSequenceClassification.from_pretrained(
        model_path, num_labels=2, ignore_mismatched_sizes=True, quantization_config=quantization_config,
        device_map="auto",
        local_files_only=True
    )
    # 同步模型的 config 中的 pad_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    # 准备模型支持 k-bit 训练
    model = prepare_model_for_kbit_training(model)
    # 配置 LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
except Exception as e:
    logging.error(f"模型加载失败: {e}")
    raise
logging.info("模型加载完成")

# 预处理数据
def tokenize_function(examples):
    return tokenizer(examples["cleaned_review"], truncation=True, padding="max_length", max_length=512)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# 将 label 转换为 0 和 1
train_dataset = train_dataset.map(lambda x: {"labels": int(x["label"])})
val_dataset = val_dataset.map(lambda x: {"labels": int(x["label"])})
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# 设置 DeepSpeed 配置文件
deepspeed_config = {
    "fp16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"}
    },
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 4
}
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

# 设置训练参数
training_args = TrainingArguments(
    output_dir="/gemini/code/finetuned_deepseek_qwen_7b",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    save_steps=100,  # 保持 100，更频繁保存
    save_total_limit=2,
    logging_steps=100,
    learning_rate=2e-5,
    eval_strategy="steps",
    eval_steps=100,  # 从 500 调整为 100，与 save_steps 匹配
    logging_dir='/gemini/code/finetune_deepseek_qwen_7b_logs',
    deepspeed=deepspeed_config,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    fp16=True,
    half_precision_backend="deepspeed"
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
model.save_pretrained("/gemini/code/finetuned_deepseek_qwen_7b")
tokenizer.save_pretrained("/gemini/code/finetuned_deepseek_qwen_7b")
logging.info("微调完成，模型保存至 /gemini/code/finetuned_deepseek_qwen_7b")