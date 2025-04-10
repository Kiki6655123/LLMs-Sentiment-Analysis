import pandas as pd

# 加载伪标签和外部语料
pseudo_df = pd.read_csv("/gemini/code/pseudo_tweets_m2m100.tsv", sep="\t")
external_df = pd.read_csv("/gemini/code/news-commentary-v18.en-zh.tsv", sep="\t", names=['source', 'target'])

# 合并
combined_df = pd.concat([pseudo_df, external_df], ignore_index=True)
combined_df.to_csv("/gemini/code/combined_corpus.tsv", sep="\t", index=False)
print(f"合并数据集完成，保存至 /gemini/code/combined_corpus.tsv，数据量: {len(combined_df)}")