#!/usr/bin/python

import jieba
import numpy as np
import sys
import logging
import logging.handlers

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class cosSim:
    def __init__(self, logger=None):
        if logger is None:
            self.logger = logging.getLogger('cos_sim')
            self.logger.setLevel(logging.DEBUG)
            # 设置日志
            formater = logging.Formatter('%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s')
            fh = logging.StreamHandler()  # 输出到终端
            fh.setLevel(level=logging.DEBUG)
            fh.setFormatter(formater)
            self.logger.addHandler(fh)

            rh = logging.handlers.TimedRotatingFileHandler('log.txt', when='D', interval=1, backupCount=30)
            rh.setLevel(level=logging.INFO)
            rh.setFormatter(formater)
            rh.suffix = "%Y%m%d_%H%M%S"
            self.logger.addHandler(rh)
        else:
            self.logger = logger

    def cos_sim(self, vector_a, vector_b):
        """ 
        计算两个向量之间的余弦相似度
        :param vector_a: 向量 a 
        :param vector_b: 向量 b
        :return: sim
        """
        vector_a = np.mat(vector_a)
        vector_b = np.mat(vector_b)
        num = float(vector_a * vector_b.T)
        denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
        cos = num / denom
        sim = 0.5 + 0.5 * cos
        return sim

    def getVocabulary(self, corpuss):
        vectorizer = CountVectorizer(max_features=500)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpuss))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        words = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
        self.logger.debug("words %s", words)
        return words

    def getVector(self, corpus, vocabulary):
        self.logger.debug("corpus %s", corpus)
        vectorizer = CountVectorizer(vocabulary=vocabulary)  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
        transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
        self.logger.debug("tf矩阵 %s", vectorizer.fit_transform(corpus))
        tfidf = transformer.fit_transform(
            vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
        weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
        # weight = sorted(weight[0], reverse=True)
        self.logger.debug("weight %s", weight)
        return weight

    def CalcuSim(self, texts=[]):
        """
        @:param list 需要对比的文本
        """
        if len(texts) != 2:
            raise Exception("texts长度必须为2")
        corpuss = [" ".join(jieba.cut(text)) for text in texts]
        vocabulary = self.getVocabulary(corpuss)
        v = self.getVector(corpuss, vocabulary=vocabulary)
        return self.cos_sim(v[0], v[1])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("error usage")
    cosSim = cosSim()
    texts = [item for item in sys.argv[1:]]
    texts = [
        "中信银行世界杯定制专属白金卡",
        "工商银行世界杯定制专属白金卡",
        "四合院型建筑大概是不少人梦寐以求的建筑别墅了，一般自建房都是农家楼房，再别致一点的就是北欧式的，应该很少有人建四合院式的吧。但是说真的，个人觉得四合院式的房子真的算得上是大气美观，实用性还超强的一款建筑房。但是这首要的就是你有足够大的建筑地",
        "但是说真的，个人觉得四合院式的房子真的算得上是大气美观，实用性还超强的一款建筑房。但是这首要的就是你有足够大的建筑地。四合院型建筑大概是不少人梦寐以求的建筑别墅了，一般自建房都是农家楼房，再别致一点的就是北欧式的，应该很少有人建四合院式的吧。但是说真的，个人觉得四合院式的房子真的算得上是大气美观，实用性还超强的一款建筑房",
        "二层平面图二楼主要是居住区，四间卧室，一间套房，一间公卫，还有一间书房，带有一个阳光房露台，室内采光通风效果好，还有一个影音区，休闲娱乐都非常不错。下面是别墅多角度立体效果图",
    ]
    corpuss = [" ".join(jieba.cut(text)) for text in texts]
    vocabulary = cosSim.getVocabulary(corpuss)
    v = cosSim.getVector(corpuss, vocabulary=vocabulary)
    print(cosSim.cos_sim(v[2], v[4]))

    # VectorA = getVector([corpuss[0]], vocabulary)
    # VectorB = getVector([corpuss[1]], vocabulary)
    # # for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
    # #     print("-------这里输出第", i, u"类文本的词语tf-idf权重------")
    # #     for j in range(len(word)):
    # #         print(word[j], weight[i][j])
    # #
    # #
    # print(cos_sim(VectorA, VectorB))
