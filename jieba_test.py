import random
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

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
def preprocess_text(content_lines, string, category):
    for line in content_lines:
        try:
            segs = jieba.lcut(line)
            segs = [v for v in segs if not str(v).isdigit()]  # 去数字
            segs = list(filter(lambda x: x.strip(), segs))  # 去左右空格
            segs = list(filter(lambda x: len(x) > 1, segs))  # 长度为1的字符
            segs = list(filter(lambda x: x not in stop_words, segs))  # 去掉停用词
            string.append((" ".join(segs), category))  # 打标签
        except Exception:
            print(line)
            continue


content = [husband, wife, son, daughter]
# 分类语料并批量添加到sentences
for value in content:
    preprocess_text(value, sentences, content.index(value))

# 打乱数据集顺序，避免同类数据分布不均匀
random.shuffle(sentences)

# for sentence in sentences[:10]:
#     print(sentence[0], sentence[1])

# 抽取特征，获得词袋模型特性
vec = CountVectorizer(
    analyzer="word",
    max_features=4000
)

x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)

# 将训练向量转换为词袋模型
vec.fit(x_train)

# 定义一个朴素贝叶斯模型, 并对训练集进行训练
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

# 评估AUC值
print(classifier.score(vec.transform(x_test), y_test))








