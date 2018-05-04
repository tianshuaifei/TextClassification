#!/usr/bin/python
# -*- coding: UTF-8 -*-
import jieba
import codecs
import re
from os.path import join, dirname, abspath
PROJECT_DIR = dirname(abspath(__file__)) # 3
class sent_seg():
    def __init__(self):
        self.stopwords=self.init_stopdict()

    def init_stopdict(self):
        stop_words = PROJECT_DIR + '/library/stop.dic'
        stopwords = codecs.open(stop_words, 'r', encoding='utf8').readlines()
        stopwords = [w.strip() for w in stopwords]
        return stopwords

    def seg(self,sent):
        sent = re.sub("[0-9\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]","",sent)
        result = []
        words = jieba.cut(sent)
        for word in words:
            #if word not in self.stopwords:
            result.append(word)
        return result
sent_seg=sent_seg()

def word_count(word_list):
    result_dict={}
    for word in word_list:
        if word not in result_dict:
            result_dict[word]=0
        result_dict[word]+=1
    counter_list = sorted(result_dict.items(), key=lambda x: x[1], reverse=True)

 #   result_dict = {k: v for k, v in result_dict.iteritems() if v >2}
    for key in list(result_dict.keys()):
        if result_dict.get(key)<1:
            del result_dict[key]

    return result_dict