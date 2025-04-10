import time
import json
import pandas as pd
from datetime import datetime
import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
import re
import uuid
from snownlp import SnowNLP
import logging

# 设置日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置Chrome驱动路径
chrome_driver_path = 'D:/Program Files/Android/chromedriver-win64/chromedriver.exe'
chrome_options = Options()
user_data_dir = r'C:\Users\Lenovo\AppData\Local\Google\Chrome\User Data'
chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
chrome_options.add_argument('--profile-directory=Default')

# 初始化WebDriver
service = Service(chrome_driver_path)
driver = webdriver.Chrome(service=service, options=chrome_options)
wait = WebDriverWait(driver, 20)

def load_cookies():
    """加载Twitter/X cookies"""
    driver.get("https://x.com/")
    time.sleep(random.uniform(5, 15))
    cookies = [
        {"name": "auth_token", "value": "1196bcad107d1f758c9de5156a8325060750ea4f", "domain": ".x.com", "path": "/"},
        {"name": "gt", "value": "1898999816707031440", "domain": ".x.com", "path": "/"},
        {"name": "guest_id", "value": "v1%3A174159317962469607", "domain": ".x.com", "path": "/"},
        {"name": "twid", "value": "u%3D1871808817249624065", "domain": ".x.com", "path": "/"},
        {"name": "lang", "value": "en", "domain": ".x.com", "path": "/"},
        {"name": "night_mode", "value": "2", "domain": ".x.com", "path": "/"},
        {"name": "att", "value": "1-K93zYDZuMb1vYSIH1JjJGPY1GloCHAUBgjJYmG5e", "domain": ".x.com", "path": "/"},
        {"name": "ct0", "value": "3f49edfd1430849b6a92d2f73e826b445aa54c4c6f589e5732cbcd44f1a47e57dbadfcc7bb0d1a25f3cf49e587be205b60c390e8b019b0e48b0b73c314f1d099d1e13d7a56add0d6418ac74e88ef527f", "domain": ".x.com", "path": "/"},
        {"name": "guest_id_ads", "value": "v1%3A174159317962469607", "domain": ".x.com", "path": "/"},
        {"name": "guest_id_marketing", "value": "v1%3A174159317962469607", "domain": ".x.com", "path": "/"},
        {"name": "kdt", "value": "QAKeSMHW4NGqipuMqhxwRgjLuOM3cMUhAYUa1LRR", "domain": ".x.com", "path": "/"},
        {"name": "personalization_id", "value": "v1_RZQCMgmB/TXywdb0caTBAg==", "domain": ".x.com", "path": "/"}
    ]
    for cookie in cookies:
        try:
            driver.delete_cookie(cookie['name'])
            driver.add_cookie(cookie)
            logging.info(f"成功添加cookie: {cookie['name']}")
        except Exception as e:
            logging.warning(f"添加cookie {cookie['name']} 失败: {str(e)}")
    driver.refresh()
    time.sleep(random.uniform(5, 15))
    # 检查是否登录成功
    current_url = driver.current_url
    logging.debug(f"当前URL: {current_url}")
    if "login" in current_url or "x.com" not in current_url:
        logging.error("Cookies 无效，未能登录 X.com")
        return False
    else:
        logging.info("Cookies 有效，已成功登录 X.com")
        return True

# 文化特征提取（俚语词典）
slang_dict = {
    "lit": "兴奋", "savage": "野蛮", "fam": "家人", "slay": "干得漂亮", "yolo": "你只活一次",
    "bruh": "兄弟", "lol": "笑", "omg": "天哪", "cool": "酷", "trash": "垃圾",
    "666": "厉害", "哈哈": "笑", "呵呵": "嘲笑", "牛逼": "厉害", "辣鸡": "垃圾",
    "无语": "无话可说", "感动": "感动", "坑爹": "坑人", "笑死": "笑死我了",
    "乱搞": "胡乱搞", "几把用": "没用", "土狗": "贬义称呼", "难受": "不舒服"
}

# 自定义情感关键词
positive_keywords = ["厉害", "好", "没错", "赚钱", "方便", "赋能", "经济发展", "支持", "cool", "great", "awesome"]
negative_keywords = ["乱搞", "难受", "垃圾", "坑爹", "不知", "疾苦", "反对", "没用", "高潮", "几把用", "咋地", "不爱国", "无国界", "trash", "bad"]

def extract_abbreviations(text):
    """提取缩写"""
    return re.findall(r'\b[A-Za-z]{2,5}\b', text)

def extract_slangs(text):
    """提取俚语"""
    words = re.split(r'\s+', text.lower())
    return [word for word in words if word in slang_dict]

def extract_features(text):
    """提取文化特征（表情、缩写、俚语）"""
    emojis_list = [e['emoji'] for e in emoji.emoji_list(text)]
    abbreviations = extract_abbreviations(text)
    slangs = extract_slangs(text)
    return {
        'emojis': emojis_list,
        'abbreviations': abbreviations,
        'slangs': slangs
    }

def get_tweet_data(tweet_element):
    try:
        text_element = tweet_element.find_element(By.XPATH, './/div[@data-testid="tweetText"]')
        text = ""
        for child in text_element.find_elements(By.XPATH, './span | ./img'):
            if child.tag_name == 'img':
                emoji_char = child.get_attribute('alt') or ''
                text += emoji_char
            else:
                text += child.text
        text = text.strip()

        time_element = tweet_element.find_element(By.XPATH, './/time')
        timestamp = time_element.get_attribute('datetime')
        author_element = tweet_element.find_element(By.XPATH, './/div[@data-testid="User-Name"]')
        author = author_element.text.split('\n')[0]
        replies = retweets = likes = "0"
        try:
            replies = tweet_element.find_element(By.XPATH, './/div[@data-testid="reply"]').text or "0"
            retweets = tweet_element.find_element(By.XPATH, './/div[@data-testid="retweet"]').text or "0"
            likes = tweet_element.find_element(By.XPATH, './/div[@data-testid="like"]').text or "0"
        except:
            pass
        cultural_features = extract_features(text)
        logging.debug(f"提取推文: {text[:20]}... Emoji: {cultural_features['emojis']}")
        return {
            "text": text,
            "timestamp": timestamp,
            "author": author,
            "engagement": {"replies": replies, "retweets": retweets, "likes": likes},
            "cultural_features": cultural_features
        }
    except Exception as e:
        logging.error(f"提取推文数据时出错: {e}")
        return None


def analyze_sentiment(text):
    try:
        detected_lang = detect(text)
        logging.debug(f"检测语言: {detected_lang}")
        if detected_lang not in ['en', 'zh']:
            detected_lang = 'en'

        sentiment_label = "neutral"
        confidence = 0.0

        if detected_lang == 'en':
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            logging.debug(f"英文情感分数: {scores}")
            compound = scores['compound']
            if compound >= 0.1:
                sentiment_label = "positive"
                confidence = compound
            elif compound <= -0.1:
                sentiment_label = "negative"
                confidence = -compound
            else:
                sentiment_label = "neutral"
                confidence = abs(compound)
        elif detected_lang == 'zh':
            text_cleaned = re.sub(r'[^\w\s]', '', text)
            logging.debug(f"清洗后文本: {text_cleaned}")
            snownlp_obj = SnowNLP(text_cleaned)
            sentiment_score = snownlp_obj.sentiments
            logging.debug(f"原始情感分数: {sentiment_score}")
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            sentiment_boost = (positive_count - negative_count) * 0.3
            logging.debug(f"正向词数: {positive_count}, 负向词数: {negative_count}, 调整值: {sentiment_boost}")
            adjusted_score = min(max(sentiment_score + sentiment_boost, 0.0), 1.0)
            logging.debug(f"调整后情感分数: {adjusted_score}")
            if adjusted_score > 0.8:
                sentiment_label = "positive"
                confidence = adjusted_score
            elif adjusted_score < 0.2:
                sentiment_label = "negative"
                confidence = 1 - adjusted_score
            else:
                sentiment_label = "neutral"
                confidence = min(abs(adjusted_score - 0.5) * 4, 1.0)

        return detected_lang, sentiment_label, confidence
    except Exception as e:
        logging.error(f"情感分析出错: {e}")
        return "unknown", "neutral", 0.0

def scroll_and_collect_tweets(query, max_scrolls=50):
    try:
        driver.get(query)
        time.sleep(random.uniform(5, 15))  # 停顿时间改为 5-15 秒
        driver.execute_script("return document.readyState === 'complete';")
        logging.debug(f"访问查询: {query}")
        tweets_collected = set()
        last_height = driver.execute_script("return document.body.scrollHeight")
        scroll_count = 0

        while scroll_count < max_scrolls:
            try:
                wait.until(EC.presence_of_all_elements_located((By.XPATH, '//article[@data-testid="tweet"]')))
                tweets = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')
                logging.debug(f"找到 {len(tweets)} 条推文")
                for tweet in tweets:
                    tweet_data = get_tweet_data(tweet)
                    if tweet_data and tweet_data['text'] not in tweets_collected:
                        tweets_collected.add(tweet_data['text'])
                        yield tweet_data
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(random.uniform(5, 15))
                new_height = driver.execute_script("return document.body.scrollHeight")
                if new_height == last_height:
                    logging.info("页面高度未变化，停止滚动")
                    break
                last_height = new_height
                scroll_count += 1
            except Exception as e:
                logging.error(f"滑动收集时出错: {e}")
                break
    except Exception as e:
        logging.error(f"访问查询 {query} 时出错: {e}")
        return

# 主题关键词列表
topics_keywords = {
    "Relationships": {
        "en": ["love", "heartbreak", "dating", "marriage", "friendship", "divorce", "cheating", "trust issues"],
        "emojis": ["❤️", "💔", "💑", "💍", "👫", "💏", "😢", "💞"]
    },
    "Mental Health": {
        "en": ["depression", "anxiety", "self-care", "therapy", "loneliness", "suicide prevention", "PTSD", "stress"],
        "emojis": ["😔", "💙", "🧠", "💆‍♂️", "🌿", "😢", "😞", "😭"]
    },
    "Social Issues": {
        "en": ["racism", "gender equality", "feminism", "LGBTQ rights", "bullying", "human rights", "poverty", "homelessness"],
        "emojis": ["✊", "♀️", "🏳️‍🌈", "💔", "⚖️", "🕊️", "🤝", "😡"]
    },
    "Politics & Controversy": {
        "en": ["elections", "war", "violence", "protests", "discrimination", "justice", "corruption", "government"],
        "emojis": ["⚖️", "🗳️", "🔥", "🚔", "💬", "🤬", "😡", "🏛️"]
    },
    "Work & Stress": {
        "en": ["burnout", "job loss", "career growth", "work-life balance", "unemployment", "toxic workplace", "overwork"],
        "emojis": ["💼", "😓", "📉", "🔥", "💪", "🧘", "😤", "😭"]
    },
    "Entertainment & Pop Culture": {
        "en": ["celebrity gossip", "scandal", "music", "movies", "fan wars", "drama", "cancel culture"],
        "emojis": ["🎬", "🎤", "🌟", "🔥", "😱", "💣", "🎭", "😡"]
    },
    "Online & Social Media": {
        "en": ["cancel culture", "hate speech", "trolling", "viral trends", "influencers", "cyberbullying", "online harassment"],
        "emojis": ["💻", "📱", "🔥", "💬", "🙄", "😡", "😠", "🛑"]
    },
    "Family & Parenting": {
        "en": ["parenting", "childhood trauma", "family conflict", "motherhood", "fatherhood", "abuse", "child neglect"],
        "emojis": ["👶", "👨‍👩‍👧‍👦", "💞", "😭", "😡", "🍼", "💔", "😢"]
    },
    "Health & Well-being": {
        "en": ["COVID-19", "vaccine", "chronic illness", "weight loss", "body positivity", "eating disorders", "disability"],
        "emojis": ["💉", "😷", "🩺", "🏋️", "🥗", "💪", "🤕", "🤒"]
    },
    "Violence & Crime": {
        "en": ["gun violence", "domestic abuse", "sexual harassment", "kidnapping", "assault", "murder", "human trafficking"],
        "emojis": ["🔫", "🆘", "🚔", "💀", "⚖️", "😡", "😭", "👮"]
    },
    "Financial Struggles": {
        "en": ["debt", "bankruptcy", "poverty", "inflation", "economic crisis", "job insecurity", "high cost of living"],
        "emojis": ["💰", "📉", "💸", "🏦", "😞", "😭", "📉", "😠"]
    },
    "Education & Student Life": {
        "en": ["exam stress", "student debt", "bullying", "peer pressure", "dropout", "mental pressure", "academic failure"],
        "emojis": ["📚", "✏️", "💭", "😭", "😓", "😞", "📖", "💡"]
    },
    "Technology & AI": {
        "en": ["AI ethics", "privacy concerns", "data security", "automation", "job displacement", "social media addiction"],
        "emojis": ["🤖", "🔒", "💾", "📡", "📱", "🧠", "😵", "🔍"]
    },
    "Addiction & Recovery": {
        "en": ["drug addiction", "alcoholism", "rehabilitation", "smoking", "gambling", "internet addiction", "relapse"],
        "emojis": ["🍷", "🚬", "💊", "💔", "😞", "😭", "🚑", "🔄"]
    },
    "Discrimination & Prejudice": {
        "en": ["ageism", "sexism", "ableism", "xenophobia", "homophobia", "transphobia", "religious discrimination"],
        "emojis": ["⚖️", "♀️", "🏳️‍🌈", "🚫", "🛑", "😡", "🤬", "😢"]
    },
    "Personal Growth & Self-Improvement": {
        "en": ["motivation", "failure", "resilience", "self-esteem", "goal setting", "mental discipline"],
        "emojis": ["🌱", "💪", "📈", "🧘", "🎯", "🔄", "🏆", "✨"]
    }
}
languages = ["en"]

def get_tweets_by_topic_and_language(topics_keywords, languages):
    all_tweets = []
    if not load_cookies():
        logging.error("登录失败，停止采集")
        return all_tweets
    for topic, keywords in topics_keywords.items():
        for lang in languages:
            query_keywords = keywords[lang] + keywords["emojis"]
            for keyword in query_keywords:
                query = f"https://x.com/search?q={keyword}%20lang%3A{lang}%20-is%3Aretweet%20-is%3Areply&src=typed_query&f=live"
                logging.info(f"正在采集: 主题={topic}, 语言={lang}, 关键词={keyword}")
                tweet_count = 0
                try:
                    for tweet_data in scroll_and_collect_tweets(query, max_scrolls=5):
                        if tweet_data:
                            detected_lang, sentiment_label, confidence = analyze_sentiment(tweet_data['text'])
                            tweet_data.update({
                                "id": f"tweet_{uuid.uuid4().hex}",
                                "topic": topic,
                                "query_language": lang,
                                "language": {"original": lang, "detected": detected_lang, "contains_mixed": detected_lang != lang},
                                "sentiment": {"overall": sentiment_label, "confidence": confidence},
                                "metadata": {
                                    "annotator": "auto",
                                    "annotation_time": datetime.now().strftime('%Y-%m-%d'),
                                    "review_status": "pending"
                                }
                            })
                            total_engagement = sum(
                                int(tweet_data['engagement'][k]) for k in ['replies', 'retweets', 'likes'] if tweet_data['engagement'][k].isdigit())
                            tweet_data['engagement']['total_engagement'] = total_engagement
                            all_tweets.append(tweet_data)
                            tweet_count += 1
                            if tweet_count >= 50:
                                break
                except Exception as e:
                    logging.error(f"采集推文时出错: 主题={topic}, 语言={lang}, 关键词={keyword}, 错误={e}")
                logging.info(f"完成采集: 主题={topic}, 语言={lang}, 关键词={keyword}, 共{tweet_count}条")
                time.sleep(random.uniform(2, 5))
    return all_tweets

def save_data(tweets_data, base_filename='twitter_data'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'total_tweets': len(tweets_data),
            'language_distribution': {},
            'topic_distribution': {}
        },
        'tweets': tweets_data
    }
    for tweet in tweets_data:
        lang = tweet['language']['detected']
        topic = tweet['topic']
        output_data['metadata']['language_distribution'][lang] = output_data['metadata']['language_distribution'].get(lang, 0) + 1
        output_data['metadata']['topic_distribution'][topic] = output_data['metadata']['topic_distribution'].get(topic, 0) + 1

    json_filename = f'{base_filename}_{timestamp}.json'
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    simplified_data = [
        {
            "label": {"positive": 1, "negative": 0, "neutral": 2}[tweet['sentiment']['overall']],
            "cleaned_review": re.sub(r'[\n\r]+', ' ', tweet['text']).strip()
        } for tweet in tweets_data
    ]
    df = pd.DataFrame(simplified_data)
    csv_filename = f'{base_filename}_{timestamp}_simplified.csv'
    df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

    return json_filename, csv_filename

# 执行采集
try:
    tweets = get_tweets_by_topic_and_language(topics_keywords, languages)
    json_filename, csv_filename = save_data(tweets)
    logging.info(f"数据已保存为 {json_filename} 和 {csv_filename}")
except Exception as e:
    logging.error(f"执行采集时出错: {e}")
finally:
    driver.quit()