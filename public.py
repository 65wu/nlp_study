import random
import jieba
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import gensim
from gensim.models import Word2Vec
from sklearn.preprocessing import scale
import multiprocessing


dir_file = "Users/Downloads/"
husband_file = "killed_by_husband.csv"
wife_file = "killed_by_wife.csv"
son_file = "killed_by_son.csv"
daughter_file = "killed_by_daughter.csv"


# 路径添加函数
def add_file(s: str):
    s = "".join([dir_file, s])
    return s


# 加载停用词
stop_words = pd.read_csv(
    add_file("stopwords.txt"),
    index_col=False,
    quoting=3,
    sep="\t",
    names=["stop_word"],
    encoding="utf-8"
)

stop_words = stop_words["stop_word"].values


# 加载语料函数、删除nan行与转换
def df_read_csv(s: str):
    unknown_df = pd.read_csv(add_file(s), names=["segment"], encoding="utf-8", sep=",")
    unknown_df.dropna(inplace=True)
    return unknown_df.segment.values.tolist()


# 加载语料
husband = df_read_csv(husband_file)
wife = df_read_csv(wife_file)
son = df_read_csv(son_file)
daughter = df_read_csv(daughter_file)

sentences = []


# 定义分词和添加标签的函数preprocess_text
# 参数content_lines为上方加载好的语料list
# 参数sentences是暂时定义的空list
# 参数category为类型标签
def preprocess_text(content_lines, string, transfer, category=0):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stop_words, segs))  # 去掉停用词
            # 分类需要提供标签
            if transfer == "classification":
                string.append((" ".join(segs), category))  # 打标签
            # 聚类则无需提供标签
            if transfer == "cluster":
                string.append(" ".join(segs))
        except Exception:
            print(line)
            continue

# for sentence in sentences[:10]:
#     print(sentence[0], sentence[1])
