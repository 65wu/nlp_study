from public import *

content = [husband, wife, son, daughter]
# 分类语料并批量添加到sentences
for value in content:
    preprocess_text(value, sentences, "classification", content.index(value))

# 打乱数据集顺序，避免同类数据分布不均匀
random.shuffle(sentences)

# 抽取特征，获得词袋模型特性
vec = CountVectorizer(
    analyzer="word",
    ngram_range=(1, 4),
    max_features=20000
)

x, y = zip(*sentences)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1256)

# 将训练向量转换为词袋模型
vec.fit(x_train)

# 定义一个朴素贝叶斯模型, 并对训练集进行训练
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)

# 评估AUC值
# print(classifier.score(vec.transform(x_test), y_test))

# 使用SVM训练
svm = SVC(kernel="linear")
svm.fit(vec.transform(x_train), y_train)
print(svm.score(vec.transform(x_test), y_test))

xgb_train = xgb.DMatrix(vec.transform(x_train), label=y_train)
xgb_test = xgb.DMatrix(vec.transform(x_test))

params = {
    "booster": "gbtree",
    "objective": "multi:softmax",
    "eval_metric": "merror",
    "num_class": 4,
    "gamma": 0.1,
    "max_depth": 8,
    "alpha": 0,
    "lambda": 10,
    "subsample": 0.7,
    "colsample_bytree": 0.5,
    "min_child_weight": 3,
    "silent": 0,
    "eta": 0.03,
    "seed": 1000,
    "nthread": -1,
    "missing": 1
}







