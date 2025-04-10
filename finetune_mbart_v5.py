import pandas as pd
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch
import nltk
import os
import emoji
import logging
from datasets import Dataset
from torch.optim import AdamW
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置 NLTK 数据路径
nltk.data.path.append('/root/nltk_data')
for resource in ['punkt', 'punkt_tab']:
    if not os.path.exists(f'/root/nltk_data/tokenizers/{resource}'):
        raise FileNotFoundError(f"NLTK 资源 {resource} 未找到")

# 预处理函数
def preprocess_emoji(text):
    emoji_dict = {'👍': ' positive ', '👎': ' negative ', '😊': ' happy ', '❤️': ' love ', '🌟': ' star ', '😃': ''}
    for emoji_char, replacement in emoji_dict.items():
        text = text.replace(emoji_char, replacement)
    text = emoji.replace_emoji(text, replace='')
    return text

def preprocess_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = preprocess_emoji(text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 加载和准备数据集
def prepare_dataset(corpus_path, document_path, sample_size=20000):
    logging.info("加载 cleaned_news_commentary 数据集...")
    news_df = pd.read_csv(corpus_path, sep='\t', names=['source', 'target'], on_bad_lines='skip')
    news_df = news_df.sample(n=sample_size, random_state=42)  # 随机抽样 20000 条
    news_df['source'] = news_df['source'].apply(preprocess_text)
    news_df['target'] = news_df['target'].apply(preprocess_text)

    logging.info("从文档中提取平行语料...")
    with open(document_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    doc_parallel_data = []
    for line in lines:
        line = line.strip()
        if line.startswith('<DOCUMENT>') or line.startswith('</DOCUMENT>'):
            continue
        if '\t' in line:
            en_text, zh_text = line.split('\t', 1)
            if en_text and zh_text:
                doc_parallel_data.append({"source": preprocess_text(en_text), "target": preprocess_text(zh_text)})

    doc_df = pd.DataFrame(doc_parallel_data)

    combined_df = pd.concat([news_df, doc_df], ignore_index=True)
    logging.info(f"合并后的数据集大小: {len(combined_df)}")

    total_size = len(combined_df)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size

    train_data = {"source": combined_df['source'][:train_size].tolist(), "target": combined_df['target'][:train_size].tolist()}
    val_data = {"source": combined_df['source'][train_size:train_size + val_size].tolist(), "target": combined_df['target'][train_size:train_size + val_size].tolist()}
    test_data = {"source": combined_df['source'][train_size + val_size:].tolist(), "target": combined_df['target'][train_size + val_size:].tolist()}

    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)

    combined_df.to_csv('/gemini/code/combined_parallel_corpus_20000.csv', index=False)
    test_dataset.save_to_disk('/gemini/code/test_dataset_20000')
    logging.info("合并后的平行语料已保存至 /gemini/code/combined_parallel_corpus_20000.csv")
    logging.info("测试数据集已保存至 /gemini/code/test_dataset_20000")

    tokenizer = MBart50TokenizerFast.from_pretrained("/gemini/code/models/mbart-large-50-many-to-many-mmt")
    return train_dataset, val_dataset, test_dataset, tokenizer

# 数据集预处理
def preprocess_function(examples, tokenizer, max_length=512):
    inputs = examples["source"]
    targets = examples["target"]
    tokenizer.src_lang = "en_XX"
    tokenizer.tgt_lang = "zh_CN"
    model_inputs = tokenizer(inputs, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 微调 mBART
def finetune_mbart(train_dataset, val_dataset, model_path="/gemini/code/models/mbart_finetuned_v3", output_dir="/gemini/code/models/mbart_finetuned_v5"):
    logging.info("加载模型和分词器...")
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_path)

    # 冻结编码器参数
    logging.info("冻结编码器参数...")
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    # 增加 dropout
    logging.info("增加 dropout...")
    model.config.dropout = 0.3  # 默认 0.1，增加到 0.3

    logging.info("预处理数据集...")
    tokenized_train = train_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)
    tokenized_val = val_dataset.map(lambda x: preprocess_function(x, tokenizer), batched=True)

    tokenized_train.save_to_disk('/gemini/code/tokenized_train_dataset_v5_20000')
    tokenized_val.save_to_disk('/gemini/code/tokenized_val_dataset_v5_20000')
    logging.info("预处理后的数据集已保存至 /gemini/code/tokenized_*_dataset_v5_20000")

    logging.info("设置训练参数...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        learning_rate=1e-5,  # 降低学习率
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,  # 减少训练轮数
        weight_decay=0.2,  # 增加正则化
        save_total_limit=2,
        save_steps=1000,
        logging_steps=500,
        fp16=True,
        use_cpu=False,
    )

    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=0.2)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        optimizers=(optimizer, None),
    )

    logging.info("开始微调...")
    trainer.train()

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"微调模型已保存至 {output_dir}")
    return model, tokenizer

if __name__ == "__main__":
    logging.info(f"GPU 可用: {torch.cuda.is_available()}")
    document_path = '/gemini/code/document.txt'
    corpus_path = '/gemini/code/cleaned_news_commentary.tsv'

    with open(document_path, 'w', encoding='utf-8') as f:
        f.write("""<DOCUMENT>
Countries, companies, and others worldwide have committed to eliminating their net greenhouse-gas emissions by a particular date – for some, as early as 2030.	世界各地的国家、企业和其他国家都承诺要在某个特定日期前消除温室气体净排放 — — 某些国家的设定早到2030年。
A 2021 report by the International Energy Agency, for example, charts a detailed path, divided into five-year intervals, toward achieving net-zero emissions by 2050 – and giving the world “an even chance of limiting the global temperature rise to 1.5°C.”	比如国际能源署于2021年发表的一份报告就描绘了一条以五年为间隔的详细路径，计划在2050年实现净零排放，并给世界“一个均等的机会去将全球气温上升幅度控制在1.5°C以内 ” 。
The most striking feature of this analysis, at least to me, is the magnitude of the decline that is required by 2030: roughly eight billion tons of fossil-fuel-based emissions, taking us from the 34 gigatons carbon dioxide today to 26 Gt.	这篇分析最显著的要点 — — 至少在我看来 — — 是到2030年所需的减排幅度：大约80亿吨化石燃料相关排放，使我们从当前的340亿吨二氧化碳排放减少到260亿吨。
If the global economy grows at a conservatively estimated annual rate of 2% over that period, the global economy’s carbon intensity (CO2 emissions per $1,000 of GDP) would need to decline by 7.8% per year.	如果全球经济在此期间以保守估计的2%年增长率增长，全球经济的碳强度（每创造1000美元GDP的二氧化碳排放量）需要每年下降7.8 % 。
While carbon intensity has been declining over the last 40 years, the trend has been nowhere near this rate: from 1980 to 2021, carbon intensity fell by just 1.3% per year, on average.	虽然该强度在过去40年间一直在下降，但这一趋势远未达到上述速度：1980~2021年间的碳强度平均每年仅下降1.3 % 。
The decline that occurred was largely a byproduct of emerging economies becoming wealthier. (More developed economies have lower carbon intensities.)	这种下降在很大程度上是新兴经济体变得更加富裕的副产品（越发达的经济体碳强度越低 ） 。
To be sure, as climate change gained more attention from policymakers, the rate of decline did accelerate, averaging 1.9% per year since 2010.	可以肯定的是，随着气候变化得到政策制定者的更多关注，下降速度确实加快了，自2010年以来平均每年下降1.9 % 。
And with supply-side constraints now encumbering the global economy – annual growth could well run at just 2% in the next few years – a modest further reduction in carbon intensity could be enough to put the global economy at or near the peak of its total CO2 emissions.	鉴于各类供给侧限制如今困扰着全球经济 — — 未来几年的年增长率很可能只有2 % — —碳排放强度的进一步小幅降低或许足以使全球经济达到或接近其二氧化碳排放总量峰值。
Higher global growth might not even set back efforts to reduce the economy’s carbon intensity, if it is fueled by the proliferation of digital technologies.	如果是由数字技术的扩散推动的话，更高的全球增长甚至可能不会阻碍降低经济碳强度的努力 ， 。
Or is it worse to acquiesce to the consequences of abandoning the ambitious path, including the risk of crossing irreversible tipping points?	或者默默接受放弃这条雄心勃勃路径的所有后果 — — 包括跨越不可逆转临界点的风险 — — 是不是更糟糕？
Half of global greenhouse-gas emissions come from just seven economies: China, the United States, the European Union, Japan, India, Canada, Australia, and Russia. The G20 economies account for 70%.	全球温室气体排放量半数来自于7个经济体：中国、美国、欧盟、日本、印度、加拿大，澳大利亚和俄罗斯，而G20经济体则占据了70 % 。
The tendency is either excessive restraint (Europe) or a diffusion of the effort (the United States).	目前的趋势是，要么是过度的克制（欧洲 ） ， 要么是努力的扩展（美国 ） 。
</DOCUMENT>""")
    logging.info("文档已保存至 /gemini/code/document.txt")

    logging.info("准备数据集...")
    train_dataset, val_dataset, test_dataset, tokenizer = prepare_dataset(corpus_path, document_path)

    logging.info("开始微调 mBART...")
    finetuned_model, finetuned_tokenizer = finetune_mbart(train_dataset, val_dataset)