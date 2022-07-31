import pandas as pd
import os
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import Perceptron
import matplotlib.pyplot as plt


def get_data():
    """
    从sklearn中获取20newspaper数据集，并将其分为训练集和测试集
    """
    newsgroup_train = fetch_20newsgroups( subset='train', categories=None, shuffle=True, random_state=12 )
    newsgroup_test = fetch_20newsgroups( subset='test', categories=None, shuffle=True, random_state=12 )
    return newsgroup_train, newsgroup_test


def tokenization(data, target, filename):
    # 字母小写化
    data = [passage.lower() for passage in data]
    # 分词
    data = [word_tokenize( passage ) for passage in data]
    # 去停用词
    tag_map = defaultdict( lambda: wn.NOUN )
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV
    for index, passage in enumerate( data ):
        final_words = []
        word_lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag( passage ):
            if word not in stopwords.words( 'english' ) and word.isalpha():
                word_final = word_lemmatized.lemmatize( word, tag_map[tag[0]] )
                final_words.append( word_final )
        data[index] = str( final_words )
    # 将dataframe保存到csv文件中
    dict_data = {"text": data, "label": target}
    df = pd.DataFrame( dict_data )
    df.to_csv( filename )


def tf_vectorization():
    # 最多将有5000个唯一的单词
    tfidf_vector = TfidfVectorizer( max_features=10000 )
    # fit(): 将矢量化器 / 模型拟合到训练数据，并将矢量化器 / 模型保存到变量( 返回sklearn.feature_extraction.text.TfidfVectorizer )
    # transform(): 使用fit() 的变量输出来转换验证 / 测试数据( 返回scipy.sparse.csr.csr_matrix )
    train_x = tfidf_vector.fit_transform( train_X )
    test_x = tfidf_vector.transform( test_X )
    validation_x = tfidf_vector.transform( validation_X )
    # train_x_tfidf 是一个文本矩阵，tfidf矩阵，稀疏矩阵表示法 存储每一个单词在每一文档中的tf-idf值
    # train_x.toarray() 将结果转化为稀疏矩阵
    # train_x.toarray().sum(axis = 0) 统计每个词在文档中的词频
    return train_x.toarray(), test_x.toarray(), validation_x.toarray()


def count_vectorization():
    count_vector = CountVectorizer( max_features=10000 )
    train_x = count_vector.fit_transform( train_X )
    test_x = count_vector.transform( test_X )
    return train_x.toarray(), test_x.toarray()


def my_knn(k, test_x, train_x, test_y, train_y):
    acc = 0
    for i in range( test_x.shape[0] ):
        distance = [np.linalg.norm( test_x[i] - train_x[k] ) for k in range( train_x.shape[0] )]
        indices = np.argsort( distance )
        a = [train_y[indices[_]] for _ in range( k )]
        predict = max( a, key=a.count )
        real = test_y[i]
        if predict == real:
            acc += 1
    return float( acc ) / len( test_x )


def knn_k():
    acc = []
    for k in range( 1, 13 ):
        acc.append( my_knn(k, validation_x_tfidf[:1000],train_x_tfidf[:3000],validation_Y[:1000],train_Y[:3000]) )
        print(acc)
    x = [k for k in range( 1, 13 )]
    plt.plot(x,acc)
    plt.savefig("knn_k")


def my_naive():
    """
    先验概率P(c):类别c的文章数量/文章总数
    似然函数P(wi|c):在类别c的文章中单词wi出现的总次数/在类别为c的文章中所有单词的总数
    """
    p_c = [train_Y.count( i ) / 8000 for i in range( 20 )]
    # print(p_c) [341, 421, 438, 398, 408, 408, 399, 429, 431, 416, 427, 420, 404, 417, 429, 440, 385, 399, 338, 252]
    # 将每一个类别的文章打包后按列求和压缩成行，count_wi_c中每一行代表在一个类别中单词wi出现的总次数。
    count_wi_c = [np.sum( [train_x_count[j] for j in range( len( train_Y ) ) if train_Y[j] == i], axis=0 ) for i in
                  range( 20 )]
    count_wi_c = np.array( count_wi_c ).__add__( 1 )
    # count_w_c代表每个列中词的总数
    count_w_c = np.sum( count_wi_c, axis=1 )
    count_w_c = np.array( count_w_c ).__add__( 10000 )
    p_wi_c = [count_wi_c[i] / count_w_c[i] for i in range( 20 )]  # (20,10000)
    acc = 0
    for i in range( len( test_Y ) ):
        p_c_d = []
        for j in range( 20 ):
            p_c_d.append( np.log2( p_c[j] ) + np.dot( np.log2( p_wi_c[j] ), test_x_count[i] ) )
        predict = p_c_d.index( max( p_c_d ) )
        if predict == test_Y[i]:
            acc += 1
    return float( acc ) / len( test_Y )


def my_perceptron(alpha=0.1,time=40000):
    acc = Perceptron.perceptron( train_x_tfidf, train_Y, test_x_tfidf, test_Y, 20, alpha=alpha,time=time)
    return acc


# 从数据集中获取训练集和测试集
train_data, test_data = get_data()
print( len( train_data.data ) )
# 对训练集和测试集中的新闻数据进行分词、去停用词、规范化，存在csv文件中。
if not os.path.exists( "train_tokenization.csv" ):
    tokenization( train_data.data[:8000], train_data.target[:8000], "train_tokenization.csv" )
if not os.path.exists( "test_tokenization.csv" ):
    tokenization( test_data.data, test_data.target, "test_tokenization.csv" )
if not os.path.exists( "validation_tokenization.csv" ):
    tokenization( train_data.data[8000:], train_data.target[8000:], "validation_tokenization.csv" )

# 从表格中获取经过第一步预处理的数据
train_tokenized = pd.read_csv( r"train_tokenization.csv", encoding='latin-1' )
train_X = train_tokenized['text']
train_Y = list( train_tokenized['label'] )
test_tokenized = pd.read_csv( r"test_tokenization.csv", encoding='latin-1' )
test_X = test_tokenized['text']
test_Y = list( test_tokenized['label'])
validation_tokenized = pd.read_csv( r"validation_tokenization.csv", encoding='latin-1' )
validation_X = validation_tokenized['text']
validation_Y = list( validation_tokenized['label'])

print( "train_data:" + str( len( train_X ) ) )
print( "test_data:" + str( len( test_X ) ) )
# print( "validation_data:" + str( len( validation_X ) ) )

# 得到tfidf向量和count向量
train_x_tfidf, test_x_tfidf, validation_x_tfidf = tf_vectorization()
train_x_count, test_x_count = count_vectorization()

# # knn 8000 2000 6 0.698 8 0.6975
# print("knn k=6")
# print("accuracy = " + str(my_knn(8, test_x_tfidf, train_x_tfidf,test_Y,train_Y)))

# # naive 8000 2000 0.777
print("naive")
print("accuracy = " + str(my_naive()))


# print("perceptron: time=40000, alpha = 0.1")
# print("accuracy = " + str(my_perceptron()))

# knn_k()