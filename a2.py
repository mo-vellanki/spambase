#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 18:16:33 2017

@author: mohit
"""

#all the imports to be imported.
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import statistics
from time import clock
from scipy.stats import friedmanchisquare as fmcs
from stac import nonparametric_tests
acc1 = []
acc2 = []
acc3 = []
columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']
data = pd.read_csv("/home/mohit/projects/bth/ML/a2/spambase/spambase.data",  delimiter=",",names=columns)
X = data.iloc[:,0:57]
y = data['spam']

clf_NB = GaussianNB()
clf_DT = DecisionTreeClassifier()
clf_LR = LogisticRegression()

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X,y)
print(skf)
print("________________________________________________")
print("Folds\t\tNB\tDT\tLR")
print("________________________________________________")
i=1    
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    start_time = clock()
    clf_NB.fit(X_train, y_train)
    time_NB = round(clock()-start_time,3)
    pred = clf_NB.predict(X_test)
    a = round(accuracy_score(y_test, pred),3)
    acc1.append(a)
    fm_NB = f1_score(y_test, pred)
    
    start_time = clock()
    clf_DT.fit(X_train, y_train)
    time_DT = round(clock() - start_time,3)
    pred = clf_DT.predict(X_test)
    b = round(accuracy_score(y_test, pred),3)
    acc2.append(b)
    fm_DT = f1_score(y_test, pred)
    
    start_time = clock()
    clf_LR.fit(X_train, y_train)
    time_LR = (clock() - start_time,3)
    pred = clf_LR.predict(X_test)
    fm_DT = f1_score(y_test, pred)
    c = round(accuracy_score(y_test, pred),3)
    acc3.append(c)
    fm_LR = f1_score(y_test, pred)

    print(i,"\t\t",a,"\t",b,"\t",c)
    i=i+1
print("______________________________________________________")
print("avg","\t","\t",statistics.mean(acc1),statistics.mean(acc2),"\t",statistics.mean(acc3))
print("stddev","\t","\t",statistics.stdev(acc1),statistics.stdev(acc2),"\t",statistics.stdev(acc3))
print("time",time_NB,"",time_DT,"", time_LR)
print("f1scores",fm_NB,"",fm_DT,"",fm_LR)
#nonparametric_tests.friedman_test(y_test,pred)