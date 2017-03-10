# -*- coding: gb18030 -*-
"""
Created on Oct 012 16:25:35 2016

@author: miaohang
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import codecs
import os
import jieba
import re
import types
import pickle as pkl
from random import shuffle
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
from gensim import models, corpora, similarities, summarization
import gensim
import itertools
import jieba
import math
import time
path_tmp = 'temp'    # 存放中间结果的位置
path_train_cut = 'data/train_cut_data' #存放分割好的训练集
path_test_cut = 'data/test_cut_data'   #存放分割好的测试集
path_predictor = 'temp/predictor' #存放分类器模型
path_corpus = 'temp/corpus_l5'    #存放预料向量
path_dictionary = os.path.join(path_tmp, 'dictionary0_l5.dict')
path_tmp_tfidfmodel = os.path.join(path_tmp, 'tf10idf2_model_l5.pkl')

test_list = os.listdir(path_test_cut)
train_list = os.listdir(path_train_cut)
corpus1_list = ['corpus_test_1', 'corpus_test_2', 'corpus_test_3', 'corpus_test_4']
corpus2_list = ['corpus_train_1', 'corpus_train_2', 'corpus_train_3', 'corpus_train_4']

start = time.clock()
if not os.path.exists(path_tmp):
    os.makedirs(path_tmp)
if not os.path.exists(path_predictor):
    os.makedirs(path_predictor)
if not os.path.exists(path_corpus):
    os.makedirs(path_corpus)
if not os.path.exists(path_train_cut):
    os.makedirs(path_predictor)
if not os.path.exists(path_test_cut):
    os.makedirs(path_predictor)
def fuc(x, y):
    res = math.sqrt(math.log(y // x + 1 , 10))
    return res
def fuc1(x):
    res = math.log(1 + x, 10)
    return res

def change_dic(dic,filter_fre, path):
    print "The length of dic is " + str(len(dic.keys()))
    small_freq_ids = [tokenid for tokenid, docfreq in dic.dfs.items() if docfreq < filter_fre]   # 去掉词典中出现次数过少的
    dic.filter_tokens(small_freq_ids)
    dic.compactify()
    dic.save(path)
    print "The length of dic_temp is " + str(len(dic.keys()))
dic = corpora.Dictionary.load(path_dictionary)
print "The length of dic is " + str(len(dic.keys()))
#change_dic(dic, 5, os.path.join(path_tmp, 'dict_train_l5.dict'))
#----
#------------------------------存储VSM向量-------------------------------
def save_corpus(dic):
    index1 = 0
    index2 = 0
    for file in test_list:
        print file + " is on training"
        words_test = []
        test = pd.read_csv(os.path.join(path_test_cut, file), encoding = 'gb18030', header = None)
        (test_num, col0) = test.shape
        for tms in range(test_num):
            if(tms % 2000 == 0):
                print "NO: " + str(tms)
            tmp = test.iloc[tms,:]
            tmp = tmp.dropna()
            tmp = tmp.tolist()
            for w in tmp:
                if(type(w) is not types.UnicodeType):
                    tmp.remove(w)
            words_test.append(tmp)
        del test
        corpus1 = [dic.doc2bow(text) for text in words_test]      #生成语料库
        corpora.MmCorpus.serialize(os.path.join(path_corpus, corpus1_list[index1]), corpus1)
        del corpus1
        index1 += 1
        del words_test
    #corpora.MmCorpus.serialize(os.path.join(path_tmp, 'corpus_test_l1.mm'), corpus1)
    #-----------------------------------------------------------------------------------------------
    for file in train_list:
        print file + " is on training"
        words_train = []
        train = pd.read_csv(os.path.join(path_train_cut, file), encoding = 'gb18030', header = None)
        (train_num, col0) = train.shape
        for tms in range(train_num):
            if(tms % 2000 == 0):
                print "NO: " + str(tms)
            tmp = train.iloc[tms,:]
            tmp = tmp.dropna()
            tmp = tmp.tolist()
            for w in tmp:
                if(type(w) is not types.UnicodeType):
			        tmp.remove(w)
            words_train.append(tmp)
        corpus2 = [dic.doc2bow(text) for text in words_train]      #生成语料库
        del train
        corpora.MmCorpus.serialize(os.path.join(path_corpus, corpus2_list[index2]), corpus2)
        index2 += 1
        del words_train
        del corpus2
#save_corpus(dic)

def tfidf_model_train(dic):
    corpus_test_1 = corpora.MmCorpus(os.path.join(path_corpus, corpus1_list[0]))
    corpus_test_2 = corpora.MmCorpus(os.path.join(path_corpus, corpus1_list[1]))
    corpus_test_3 = corpora.MmCorpus(os.path.join(path_corpus, corpus1_list[2]))
    corpus_test_4 = corpora.MmCorpus(os.path.join(path_corpus, corpus1_list[3]))
    corpus_train_1 = corpora.MmCorpus(os.path.join(path_corpus, corpus2_list[0]))
    corpus_train_2 = corpora.MmCorpus(os.path.join(path_corpus, corpus2_list[1]))
    corpus_train_3 = corpora.MmCorpus(os.path.join(path_corpus, corpus2_list[2]))
    corpus_train_4 = corpora.MmCorpus(os.path.join(path_corpus, corpus2_list[3]))
    corpus1 = [x for x in corpus_test_1] + [x for x in corpus_test_2] + [x for x in corpus_test_3] + [x for x in corpus_test_4]
    corpora.MmCorpus.serialize(os.path.join(path_corpus, 'corpus_test'), corpus1)
    corpus2 = [x for x in corpus_train_1] + [x for x in corpus_train_2] + [x for x in corpus_train_3] + [x for x in corpus_train_4]
    corpora.MmCorpus.serialize(os.path.join(path_corpus, 'corpus_train'), corpus2)
    merged_corpus = [x for x in corpus1] + [x for x in corpus2]
    tfidf = models.TfidfModel(dictionary=dic,
                          corpus = merged_corpus,
                          wglobal=fuc)                          #生成tfidf向量
    tfidf_file = open(path_tmp_tfidfmodel, 'wb')               #存储tf_idf模型
    pkl.dump(tfidf, tfidf_file)
    tfidf_file.close()
    print "TF-IDF model is done"
    end = time.clock()
    print "cost time is: %f s" % (end - start)
tfidf_model_train(dic)
