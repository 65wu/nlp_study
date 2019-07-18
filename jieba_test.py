import random
import jieba
import pandas as pd


dir_file = "Users/Downloads/"
husband_file = "killed_by_husband.csv"
wife_file = "killed_by_wife.csv"
son_file = "killed_by_son.csv"
daughter_file = "killed_by_husband.csv"


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



