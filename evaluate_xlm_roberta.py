import pandas as pd
from transformers import AutoTokenizer, XLMRobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import logging
import os
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# 设置日志
log_file = f'/gemini/code/evaluate_xlm_roberta_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', encoding='utf-8')
logging.info("开始评估 xlm-roberta-base 和 finetuned_xlm_roberta 对比")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"使用设备: {device}")
if torch.cuda.is_available():
    logging.info(f"GPU 名称: {torch.cuda.get_device_name(0)}, 显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted")
    }

def load_specific_dataset(file_paths):
    dfs = []
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logging.error(f"数据集文件不存在: {file_path}")
            raise FileNotFoundError(f"数据集文件不存在: {file_path}")
        df = pd.read_csv(file_path)
        df = df[df["label"].isin([0, 1])]  # 过滤 label=2
        label_counts = df["label"].value_counts().to_dict()
        logging.info(f"{file_path} label 分布（过滤后）: {label_counts}")
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    logging.info(f"加载了 {len(dfs)} 个文件，共 {len(combined_df)} 条数据（已过滤 label=2）")
    return Dataset.from_pandas(combined_df)

def load_model_and_tokenizer(base_path, finetuned_path=None):
    try:
        logging.info("开始加载 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(finetuned_path or base_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logging.info(f"未定义填充标记，使用 EOS 标记: {tokenizer.pad_token}")
        logging.info(f"Tokenizer 配置: pad_token={tokenizer.pad_token}, pad_token_id={tokenizer.pad_token_id}")
        
        logging.info("开始加载基础模型...")
        base_model = XLMRobertaForSequenceClassification.from_pretrained(
            base_path,
            num_labels=2,
            device_map="auto",  # 自动分配设备
            ignore_mismatched_sizes=True
        )
        base_model.config.pad_token_id = tokenizer.pad_token_id
        logging.info(f"基础模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        if finetuned_path and os.path.exists(finetuned_path):
            logging.info("开始加载微调模型...")
            model = XLMRobertaForSequenceClassification.from_pretrained(
                finetuned_path,
                num_labels=2,
                device_map="auto",
                ignore_mismatched_sizes=True
            )
            model.config.pad_token_id = tokenizer.pad_token_id
            logging.info(f"微调模型加载完成，显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        else:
            model = base_model
        return model, tokenizer
    except Exception as e:
        logging.error(f"模型加载失败: {e}")
        raise

def tokenize_function(examples, tokenizer):
    texts = [str(text) if text is not None else "" for text in examples["cleaned_review"]]
    return tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
        pad_to_max_length=True
    )

def preprocess_dataset(dataset):
    possible_columns = ["cleaned_review", "cleaned_text", "text"]
    for col in possible_columns:
        if col in dataset.column_names:
            logging.info(f"使用列名: {col}")
            if col == "cleaned_review":
                return dataset
            return dataset.rename_column(col, "cleaned_review")
    raise ValueError("未找到合适的文本列名，请检查数据集")

def validate_dataset(dataset, text_column="cleaned_review"):
    invalid_text_count = sum(1 for ex in dataset if not isinstance(ex[text_column], str) or ex[text_column] is None)
    if invalid_text_count > 0:
        logging.warning(f"发现 {invalid_text_count} 条无效文本记录，将转换为空字符串")
    invalid_label_count = sum(1 for ex in dataset if not isinstance(ex["label"], (int, float)) or ex["label"] not in [0, 1])
    if invalid_label_count > 0:
        logging.warning(f"发现 {invalid_label_count} 条无效标签记录，将修正为 0")
    dataset = dataset.map(lambda x: {
        text_column: str(x[text_column]) if x[text_column] is not None else "",
        "label": int(x["label"]) if isinstance(x["label"], (int, float)) and x["label"] in [0, 1] else 0
    })
    return dataset

class CustomTrainer(Trainer):
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        if not prediction_loss_only:
            loss, logits, labels = outputs
            logging.debug(f"logits shape: {logits.shape}, labels shape: {labels.shape}")
            logging.debug(f"logits sample: {logits[:5]}, labels sample: {labels[:5]}")
        return outputs

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
    
    probs = torch.softmax(torch.tensor(pred_output.predictions), dim=-1)[:, 1].numpy()
    fpr, tpr, _ = roc_curve(dataset["labels"], probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.legend()
    plt.title(f"ROC Curve - {name}")
    plt.savefig(f"/gemini/code/{name}_roc.png")
    plt.close()
    return eval_results

if __name__ == "__main__":
    data_files = [
        "/gemini/code/eval_unseen_comments_2000_v4.csv",
        "/gemini/code/enhanced_tweets.csv"
    ]
    try:
        unseen_dataset = load_specific_dataset(data_files)
        unseen_dataset = preprocess_dataset(unseen_dataset)
        unseen_dataset = validate_dataset(unseen_dataset)
        logging.info("成功加载并验证未见数据")
    except FileNotFoundError as e:
        logging.error(f"未见数据文件未找到: {e}")
        raise

    base_model_path = "/gemini/code/xlm-roberta-base"
    finetuned_model_path = "/gemini/code/finetuned_xlm_roberta"

    if not os.path.exists(base_model_path):
        logging.error(f"基础模型路径不存在: {base_model_path}")
        raise FileNotFoundError(f"基础模型路径不存在: {base_model_path}")
    if not os.path.exists(finetuned_model_path):
        logging.error(f"微调模型路径不存在: {finetuned_model_path}")
        raise FileNotFoundError(f"微调模型路径不存在: {finetuned_model_path}")

    base_model, base_tokenizer = load_model_and_tokenizer(base_model_path)
    finetuned_model, finetuned_tokenizer = load_model_and_tokenizer(base_model_path, finetuned_model_path)

    logging.info("开始预处理数据...")
    base_dataset = unseen_dataset.map(lambda x: tokenize_function(x, base_tokenizer), batched=True)
    base_dataset = base_dataset.map(lambda x: {"labels": int(x["label"])})
    base_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    finetuned_dataset = unseen_dataset.map(lambda x: tokenize_function(x, finetuned_tokenizer), batched=True)
    finetuned_dataset = finetuned_dataset.map(lambda x: {"labels": int(x["label"])})
    finetuned_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    logging.info("数据预处理完成")

    eval_args = TrainingArguments(
        output_dir="/gemini/code/eval_xlm_roberta",
        per_device_eval_batch_size=4,
        logging_dir='/gemini/code/eval_xlm_roberta_logs',
        report_to="none",
        fp16=True,
        do_train=False,
        do_eval=True
    )

    base_trainer = CustomTrainer(
        model=base_model,
        args=eval_args,
        eval_dataset=base_dataset,
        compute_metrics=compute_metrics
    )
    finetuned_trainer = CustomTrainer(
        model=finetuned_model,
        args=eval_args,
        eval_dataset=finetuned_dataset,
        compute_metrics=compute_metrics
    )

    logging.info("开始评估基础模型...")
    base_results = evaluate_and_plot(base_trainer, base_dataset, "base")

    logging.info("开始评估微调模型...")
    finetuned_results = evaluate_and_plot(finetuned_trainer, finetuned_dataset, "finetuned")

    logging.info("性能对比：")
    logging.info(f"基础模型: {base_results}")
    logging.info(f"微调模型: {finetuned_results}")
    print("\n=== 性能对比 ===")
    print(f"基础模型: {base_results}")
    print(f"微调模型: {finetuned_results}")

    metrics = ["eval_loss", "accuracy", "f1", "precision", "recall"]
    for metric in metrics:
        if metric in base_results and metric in finetuned_results:
            if metric == "eval_loss":
                improvement = (base_results[metric] - finetuned_results[metric]) / base_results[metric] * 100
                logging.info(f"{metric} 减少百分比: {improvement:.2f}%")
                print(f"{metric} 减少百分比: {improvement:.2f}%")
            else:
                improvement = (finetuned_results[metric] - base_results[metric]) / base_results[metric] * 100
                logging.info(f"{metric} 提升百分比: {improvement:.2f}%")
                print(f"{metric} 提升百分比: {improvement:.2f}%")

    logging.info("可视化结果已保存至 /gemini/code/base_cm.png 等文件")
    torch.cuda.empty_cache()
    logging.info("评估完成，GPU 内存已清理")