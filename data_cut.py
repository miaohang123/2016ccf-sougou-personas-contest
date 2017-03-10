'''
将训练集数据user_train.csv和测试集数据user_test.csv进行分割处理
目的是为了加快用gensim进行文本处理的速度

'''
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
from gensim import models, corpora, similarities, summarization
import gensim
import itertools
import jieba
import math
import time
#------------------------------------------------------------------------------
train = pd.read_csv('data/user_train.csv', encoding = 'gb18030', header = None)
test = pd.read_csv('data/user_test.csv', encoding = 'gb18030', header = None)
(train_num, col0) = train.shape
(test_num, col1) = test.shape
stopwords = codecs.open('stoplist.txt','r',encoding='utf-8').readlines()
stopwords = [w.strip() for w in stopwords ] + [' ', '']
word_flag = [u'n', u'nr', u'nr1', u'nr2', u'ns', u'nt', u'nz', u'nl', u'v', u'vd', u'vn', u'vi', u'b', u'bl', u'a', u'an', u'al', u'r', u'm', u'mq', u'l', u'j']
patt = '(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])? | \d+ | ((?:(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))\.){3}(?:25[0-5]|2[0-4]\d|((1\d{2})|([1-9]?\d)))) | /^(-|\+)?\d+$/; | /d{3}-/d{8}|/d{4}-/d{7}'
patt2 = '^[0-9]+(\\.[0-9]+)?$'
words_test1 = []
words_test2 = []
words_test3 = []
words_test4 = []
#----------------------------------------------------------------------
#------------------第一段----------------------
for tms in range(int(0.25 * test_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    test_temp = test.iloc[tms, 1:]
    test_temp = test_temp.dropna()
    query_test = test_temp.tolist()
    words_temp = []
    for doc in query_test:
        if(type(doc) is not types.UnicodeType):
            query_test.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_test1.append(listall)
test_save = pd.DataFrame(words_test1)
del words_test1
test_save.to_csv(os.path.join(path_test_cut, "test_cut1.csv"), index = False, header = False, encoding = 'gb18030')
del test_save
#---------------------第二段------------------
for tms in range(int(0.25 * test_num), int(0.5 * test_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    test_temp = test.iloc[tms, 1:]
    test_temp = test_temp.dropna()
    query_test = test_temp.tolist()
    words_temp = []
    for doc in query_test:
        if(type(doc) is not types.UnicodeType):
            query_test.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_test2.append(listall)
test_save = pd.DataFrame(words_test2)
del words_test2
test_save.to_csv(os.path.join(path_test_cut, "test_cut2.csv"), index = False, header = False, encoding = 'gb18030')
del test_save
#---------------------第三段------------------
for tms in range(int(0.5 * test_num), int(0.75 * test_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    test_temp = test.iloc[tms, 1:]
    test_temp = test_temp.dropna()
    query_test = test_temp.tolist()
    words_temp = []
    for doc in query_test:
        if(type(doc) is not types.UnicodeType):
            query_test.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_test3.append(listall)
test_save = pd.DataFrame(words_test3)
del words_test3
test_save.to_csv(os.path.join(path_test_cut, "test_cut3.csv"), index = False, header = False, encoding = 'gb18030')
del test_save
#---------------------第四段------------------
for tms in range(int(0.75 * test_num), test_num):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    test_temp = test.iloc[tms, 1:]
    test_temp = test_temp.dropna()
    query_test = test_temp.tolist()
    words_temp = []
    for doc in query_test:
        if(type(doc) is not types.UnicodeType):
            query_test.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_test4.append(listall)
test_save = pd.DataFrame(words_test4)
del words_test4
test_save.to_csv(os.path.join(path_test_cut, "test_cut4.csv"), index = False, header = False, encoding = 'gb18030')
del test_save
del test


#------------------------训练集分词-----------------------------------------
words_train1 = []
words_train2 = []
words_train3 = []
words_train4 = []
#--------------------第一段------------------
for tms in range(int(0.25 * train_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    train_temp = train.iloc[tms, 4:]
    train_temp = train_temp.dropna()
    query_train = train_temp.tolist()
    words_temp = []
    for doc in query_train:
        if(type(doc) is not types.UnicodeType):
            query_train.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_train1.append(listall)
train_save = pd.DataFrame(words_train1)
del words_train1
train_save.to_csv(os.path.join(path_train_cut, "train1_cut.csv"), header = None, index = None, encoding = "gb18030")
del train_save
#---------------------第二段-------------------
for tms in range(int(0.25 * train_num),int(0.5 * train_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    train_temp = train.iloc[tms, 4:]
    train_temp = train_temp.dropna()
    query_train = train_temp.tolist()
    words_temp = []
    for doc in query_train:
        if(type(doc) is not types.UnicodeType):
            query_train.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_train2.append(listall)
train_save = pd.DataFrame(words_train2)
del words_train2
train_save.to_csv(os.path.join(path_train_cut, "train2_cut.csv"), header = None, index = None, encoding = "gb18030")
del train_save
#---------------------第三段-------------------
for tms in range(int(0.5 * train_num),int(0.75 * train_num)):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    train_temp = train.iloc[tms, 4:]
    train_temp = train_temp.dropna()
    query_train = train_temp.tolist()
    words_temp = []
    for doc in query_train:
        if(type(doc) is not types.UnicodeType):
            query_train.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_train3.append(listall)
train_save = pd.DataFrame(words_train3)
del words_train3
train_save.to_csv(os.path.join(path_train_cut, "train3_cut.csv"), header = None, index = None, encoding = "gb18030")
del train_save
#---------------------第四段-------------------
for tms in range(int(0.75 * train_num),train_num):#(int(p * train_num)):
    if tms % 1000 == 0:
        print "No. %s" %tms
    train_temp = train.iloc[tms, 4:]
    train_temp = train_temp.dropna()
    query_train = train_temp.tolist()
    words_temp = []
    for doc in query_train:
        if(type(doc) is not types.UnicodeType):
            query_train.remove(doc)
            continue
        m = re.search(patt, doc)
        if m is None:
            doc = list(jieba.cut(doc))
            words_temp.append([unicode(w) for w in doc if (w not in stopwords) and (type(w) is types.UnicodeType) and (re.match(patt2, w) is None)])
    listall = []
    for line in words_temp:
        listall.extend(line)
    words_train4.append(listall)
train_save = pd.DataFrame(words_train4)
del words_train4
train_save.to_csv(os.path.join(path_train_cut, "train4_cut.csv"), header = None, index = None, encoding = "gb18030")
del train_save
del train

end = time.clock()
print "cost time is: %f s" % (end - start)
'''
'''
#------------------------------直接读测试集分词---------------------
test_list = os.listdir(path_test_cut)
train_list = os.listdir(path_train_cut)
dic1 = corpora.Dictionary()
dic2 = corpora.Dictionary()
for file in test_list:
    print file + " is on training"
    words_test = []
    test = pd.read_csv(os.path.join(path_test_cut, file), encoding = 'gb18030', header = None)
    (test_num, col0) = test.shape
    for tms in range(test_num):
        if(tms % 1000 == 0):
            print "NO: " + str(tms)
        tmp = test.iloc[tms,:]
        tmp = tmp.dropna()
        tmp = tmp.tolist()
        for w in tmp:
            if(type(w) is not types.UnicodeType):
                tmp.remove(w)
            #words_test.append(tmp)
        dic1.add_documents([tmp])
    #dic1.add_documents(words_test)
    del test
    del words_test
print "test data loaded successfully"
#dic1 = corpora.Dictionary(words_test)                      #生成词典
print "The numbers of the dictionary is " + str(len(dic1.keys()))
dic1.save(os.path.join(path_tmp, 'dict_test_l1.dict'))                               #存储字典
'''
'''
#------------------------------直接读训练集分词结果-------------------------------
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
            #words_train.append(tmp)
        dic2.add_documents([tmp])
    #dic2.add_documents(words_train)
    del train
    del words_train
print "train data loaded successfully"
#dic2 = corpora.Dictionary(words_train)                      #生成词典
dic2.save(os.path.join(path_tmp, 'dict_train_l1.dict'))                               #存储字典
print "The numbers of the dictionary is " + str(len(dic2.keys()))