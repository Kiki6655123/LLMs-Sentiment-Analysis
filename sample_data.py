# sampled_data.py
import pandas as pd

# 加载清洗后的数据
df = pd.read_csv("/gemini/code/cleaned_weibo_comments.csv")

# 按 label 分组，分别抽样
positive_df = df[df['label'] == 1]
negative_df = df[df['label'] == 0]

# 确保正负样本数量接近
min_count = min(len(positive_df), len(negative_df), 1000) 
sampled_positive = positive_df.sample(n=min_count, random_state=52)
sampled_negative = negative_df.sample(n=min_count, random_state=52)

# 合并抽样数据
sampled_df = pd.concat([sampled_positive, sampled_negative])

# 打乱数据
sampled_df = sampled_df.sample(frac=1, random_state=52).reset_index(drop=True)

# 保存抽样数据
sampled_df.to_csv("/gemini/code/weibo_comments_2000.csv", index=False)
print("抽样完成，保存至 /gemini/code/weibo_comments_2000.csv")