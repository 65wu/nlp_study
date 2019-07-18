from public import *

content = [husband, wife, son, daughter]
# 分类语料并批量添加到sentences
for value in content:
    preprocess_text(value, sentences, transfer="cluster")

# 打乱数据集顺序，避免同类数据分布不均匀
random.shuffle(sentences)
# 构建词频矩阵
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5)
# 统计每个词语的tf-idf权值
transformer = TfidfTransformer()
# 第一个fit_transform：计算tf-idf
# 第二个fit_transform：将文本转换成词频矩阵
tf_idf = transformer.fit_transform(vectorizer.fit_transform(sentences))
word = vectorizer.get_feature_names()
# 将tf_idf矩阵抽出
weight = tf_idf.toarray()
# 查看特征值大小
print("Feature length：" + str(len(word)))
