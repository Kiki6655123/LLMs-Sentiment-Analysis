import pandas as pd
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from tqdm import tqdm
import torch

# 加载 M2M100 模型和 tokenizer
model = M2M100ForConditionalGeneration.from_pretrained("/gemini/code/models/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("/gemini/code/models/m2m100_418M")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

# 翻译函数
def translate_m2m100(texts, src_lang="en", tgt_lang="zh", batch_size=32):
    model.eval()
    translated_texts = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"翻译 {src_lang} 到 {tgt_lang}"):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            num_beams=5,
            max_length=512,
            length_penalty=1.0,
            early_stopping=True
        )
        translated_texts.extend([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
    return translated_texts

# 加载 Twitter 数据
df = pd.read_csv("/gemini/code/filtered_tweets_sentiment.csv")
texts = df['cleaned_review'].tolist()

# 生成伪标签
zh_texts = translate_m2m100(texts, "en", "zh")
pseudo_df = pd.DataFrame({"source": texts, "target": zh_texts})
pseudo_df.to_csv("/gemini/code/pseudo_tweets_m2m100.tsv", sep="\t", index=False)
print("伪标签生成完成，保存至 /gemini/code/pseudo_tweets_m2m100.tsv")