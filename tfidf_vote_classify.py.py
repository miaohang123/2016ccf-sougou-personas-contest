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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from gensim import models, corpora, similarities
import gensim
import time
import math
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import feature_selection
from sklearn.feature_selection import chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.cross_validation import StratifiedKFold
#-----------------------------------------------------------------
path_tmp = 'temp'  
path_predictor = 'temp/predictor'
path_corpus = 'temp/corpus_l5'
path_dictionary = os.path.join(path_tmp, 'dictionary0_l5.dict')
path_tmp_tfidfmodel = os.path.join(path_tmp, 'tf10idf_model_l5.pkl')
path_tmp_predictor1 = os.path.join(path_predictor, 'SVC1_tf10idf_l5.m')
path_tmp_predictor2 = os.path.join(path_predictor, 'SGD2_tf10idf_l5.m')
path_tmp_predictor3 = os.path.join(path_predictor, 'SVC3_tf10idf_l5.m')


start = time.clock()
def fuc(x, y):
    res = math.log(y // x, 10)
    return res
def fuc1(x):
    res = math.log(1 + x, 10)
    return res

if not os.path.exists(path_predictor):
    os.makedirs(path_predictor)

print "Program started"
#======================================================================================
#======================================读取词典和tfidf模型============================================
#======================================================================================
dic = []
tfidf = []
if not dic:
    dic = corpora.Dictionary.load(path_dictionary)
    print "dictionary loaded successfully"
lenofdic = len(dic.keys())
if not tfidf:
    tfidf_file = open(path_tmp_tfidfmodel, 'rb')
    tfidf = pkl.load(tfidf_file)
    tfidf_file.close()
    print "tfidf_model loaded successfully"
print "The numbers of the dictionary is " + str(len(dic.keys()))

#------------------------------读取VSM向量---------------------------------
corpus_train_list = ['corpus_train_1', 'corpus_train_2', 'corpus_train_3', 'corpus_train_4']
corpus_train_1 = corpora.MmCorpus(os.path.join(path_corpus, corpus_train_list[0]))
corpus_train_2 = corpora.MmCorpus(os.path.join(path_corpus, corpus_train_list[1]))
corpus_train_3 = corpora.MmCorpus(os.path.join(path_corpus, corpus_train_list[2]))
corpus_train_4 = corpora.MmCorpus(os.path.join(path_corpus, corpus_train_list[3]))
train_vsm0 = [x for x in corpus_train_1] + [x for x in corpus_train_2] + [x for x in corpus_train_3] + [x for x in corpus_train_4]
print "train_vsm0 generated successfully"
#----------------------------------------------------------------
train_vsm1 = []
len_of_trainvsm = []
#---------------
for i in range(len(train_vsm0)):
    len_of_trainvsm.append(len(train_vsm0[i]))
len_of_trainvsm = np.array(len_of_trainvsm)
#---------------
for i in range(len(train_vsm0)):
    temp = []
    for j in range(len(train_vsm0[i])):
            temp.append((train_vsm0[i][j][0], 1))
    train_vsm1.append(temp)
#print "max of the length of train_vsm is: " + str(len_of_trainvsm.max())
#print "min of the length of train_vsm is: " + str(len_of_trainvsm.min())
#print "mean of the length of train_vsm is: " + str(len_of_trainvsm.mean())
#-----------------------------生成TFIDF---------------------------------------
tfidf_matrix = tfidf[train_vsm1]
print "tfidf_matrix generated successfully"
del train_vsm0
del train_vsm1
del tfidf
#----------------------------读取标签-----------------------------------
train = pd.read_csv('user_train.csv', encoding = 'gb18030', header = None)
train_label1 = train.iloc[:, 1]
train_label2 = train.iloc[:, 2]
train_label3 = train.iloc[:, 3]
del train
train_label1 = train_label1.as_matrix()
train_label2 = train_label2.as_matrix()
train_label3 = train_label3.as_matrix()
zero_lineid = []
print "train_label loaded successfully"
#-------------------------去缺失值--------------------------
for i in range(train_label1.shape[0]):#(train_lsi_matrix.shape[0]):
    if(train_label1[i] == 0 or train_label2[i] == 0 or train_label3[i] == 0):
        zero_lineid.append(i)
train_label1 = np.delete(train_label1, zero_lineid, axis = 0)
train_label2 = np.delete(train_label2, zero_lineid, axis = 0)
train_label3 = np.delete(train_label3, zero_lineid, axis = 0)
tfidf_matrix = np.delete(tfidf_matrix, zero_lineid, axis = 0)
del zero_lineid
#---------------------------打乱顺序----------------------------
print len(tfidf_matrix)
indices = np.arange(len(tfidf_matrix))
tfidf_matrix = tfidf_matrix[indices]
train_label1 = train_label1[indices]
train_label2 = train_label2[indices]
train_label3 = train_label3[indices]
del indices
#-----------------------------------gensim------------------------------------------
tfidf_sparse_matrix = gensim.matutils.corpus2csc(tfidf_matrix, num_terms = lenofdic).transpose()
del tfidf_matrix
print "The dimension of tfidf_sparse_matrix is "+ str(tfidf_sparse_matrix._shape)
#----------------------------------------------------------------------------
def Vote_train():
    clf1 = LogisticRegression(solver='sag', max_iter=150, random_state=1)
    clf2 = LogisticRegression(solver='liblinear', random_state=1)
    clf3 = SGDClassifier(loss='squared_hinge', n_iter= 25)
    eclf1 = VotingClassifier(estimators=[('lr1', clf1), ('lr2', clf2),('sgd', clf3)], voting='hard')
    eclf1.fit(tfidf_sparse_matrix, train_label1)
    joblib.dump(eclf1, path_tmp_predictor1, compress=3)
    end = time.clock()
    del clf1
    del clf2
    del clf3
    del eclf1
    #--------------------------
    model2 =SGDClassifier(alpha=2e-05, average=False, class_weight=None, epsilon=0.1,
           eta0=0.1, fit_intercept=True, l1_ratio=0.15,
        learning_rate='constant', loss='hinge', n_iter=5, n_jobs=1,
        penalty='l2', power_t=0.7, random_state=None, shuffle=True,
        verbose=0, warm_start=False)
    model2.fit(tfidf_sparse_matrix, train_label2)
    joblib.dump(model2, path_tmp_predictor2, compress=3)
    #--------------------------
    clf1 = LogisticRegression(solver='sag', max_iter=150, random_state=1)
    clf2 = LogisticRegression(solver='liblinear', random_state=1)
    clf3 = SGDClassifier(loss='squared_loss', n_iter=25)
    eclf3 = VotingClassifier(estimators=[('lr1', clf1), ('lr2', clf2), ('SGD', clf3)])
    eclf3.fit(tfidf_sparse_matrix, train_label3)
    joblib.dump(eclf3, path_tmp_predictor3, compress=3)
    end = time.clock()
    del clf1
    del clf2
    del eclf3
Vote_train()
#--------------------------------------------------------------------
print "Predicting part starts"
dic = []
tfidf = []
if not dic:
    dic = corpora.Dictionary.load(path_dictionary)
    print "dictionary loaded successfully"
lenofdic = len(dic.keys())
if not tfidf:
    tfidf_file = open(path_tmp_tfidfmodel, 'rb')
    tfidf = pkl.load(tfidf_file)
    tfidf_file.close()
    print "tfidf_model loaded successfully"
#------------------------------????????VSM??ò????---------------------------------
corpus_test_list = ['corpus_test_1', 'corpus_test_2', 'corpus_test_3', 'corpus_test_4']
corpus_test_1 = corpora.MmCorpus(os.path.join(path_corpus, corpus_test_list[0]))
corpus_test_2 = corpora.MmCorpus(os.path.join(path_corpus, corpus_test_list[1]))
corpus_test_3 = corpora.MmCorpus(os.path.join(path_corpus, corpus_test_list[2]))
corpus_test_4 = corpora.MmCorpus(os.path.join(path_corpus, corpus_test_list[3]))
test_vsm0 = [x for x in corpus_test_1] + [x for x in corpus_test_2] + [x for x in corpus_test_3] + [x for x in corpus_test_4]
print "test_vsm0 generated successfully"
#------------------------------??????????è????????1----------------------------------
test_vsm = []
for i in range(len(test_vsm0)):
	temp = []
	for j in range(len(test_vsm0[i])):
            temp.append((test_vsm0[i][j][0], 1))
	test_vsm.append(temp)
print "test_vsm generated successfully"
#----------------------------??ú????TF-IDF??ò????---------------------------------
test_tfidf_matrix = tfidf[test_vsm]
print "test_tfidf_matrix generated successfully"
del test_vsm
del test_vsm0
del tfidf
#----------------------------gensim??ò????×??????×é---------------------------------
test_tfidf_sparse_matrix = gensim.matutils.corpus2csc(test_tfidf_matrix, num_terms = lenofdic).transpose()
print "TF-IDF sparse matrix generated successfully"
print "The dimension of test_tfidf_sparse_matrix is " + str(test_tfidf_sparse_matrix._shape)
del test_tfidf_matrix
del dic
#-----------------------------??¤????-----------------------------------------------------
test_label1 = model1.predict(test_tfidf_sparse_matrix)
test_label2 = model2.predict(test_tfidf_sparse_matrix)
test_label3 = model3.predict(test_tfidf_sparse_matrix)
del test_tfidf_sparse_matrix
del model1
del model2
del model3
#------------------------------??á????????????csv---------------------------------------------
test = pd.read_csv('user_test.csv', encoding = 'gb18030', header = None)
(test_num, col0) = test.shape
test_id = test.iloc[:, 0]
test_id = test_id.as_matrix()
test_result = []   #[[id, label1, label2, label3]]
del test
for tms in range(test_num):
    temp_result = [test_id[tms], test_label1[tms], test_label2[tms], test_label3[tms]]
    test_result.append(temp_result)
del test_id
del test_label1
del test_label2
del test_label3
test_result = pd.DataFrame(test_result)
print test_result.head()
test_result.to_csv("svc_sgd_result_l5.csv", index = False, header = False, sep = " ", encoding = 'gbk')
print "The result file :svc_result generated successfully"
del test_result
end = time.clock()
print "cost time is: %f s" % (end - start)
print "Predict part is over"

