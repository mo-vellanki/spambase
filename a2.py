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
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import statistics
from time import clock
from scipy.stats import friedmanchisquare as fmcs
from stac import nonparametric_tests
import numpy as np
from scipy.stats import rankdata

acc_NB = []
acc_DT = []
acc_LR = []
fm_NB = []
fm_DT = []
fm_LR = []
t_NB = []
t_DT = []
t_LR = []


columns = ['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total', 'spam']
data = pd.read_csv("/home/mohit/projects/bth/ML/a2/spambase/spambase.data",  delimiter=",",names=columns)
X = data.iloc[:,0:57]
y = data['spam']

#clf_NB = LogisticRegression()
#clf_DT = LinearSVC()
#clf_LR = DecisionTreeClassifier()
clf_NB = GaussianNB()
clf_DT = DecisionTreeClassifier()
clf_LR = LogisticRegression()

skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(X,y)
print(skf)
print("============")
print("Table 1")
print("============")
print("____________________________________________________________________")
print("Folds\t\tNaive Bayes\tDecision Tree\tLogistic Regression")
print("____________________________________________________________________")
i=1    
for train_index, test_index in skf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    start_time = clock()
    clf_NB.fit(X_train, y_train)
    t_NB.append(round(clock()-start_time,3))
    pred = clf_NB.predict(X_test)
    a = round(accuracy_score(y_test, pred),4)
    acc_NB.append(a)
    a1 = f1_score(y_test, pred)
    fm_NB.append(a1)
    
    start_time = clock()
    clf_DT.fit(X_train, y_train)
    t_DT.append(round(clock() - start_time,3))
    pred = clf_DT.predict(X_test)
    b = round(accuracy_score(y_test, pred),4)
    acc_DT.append(b)
    b1 = f1_score(y_test, pred)
    fm_DT.append(b1)
    
    start_time = clock()
    clf_LR.fit(X_train, y_train)
    t_LR.append(round(clock() - start_time,3))
    pred = clf_LR.predict(X_test)
    c = round(accuracy_score(y_test, pred),4)
    acc_LR.append(c)
    c1 = f1_score(y_test, pred)
    fm_LR.append(c1)
    
    print(i,"\t\t%.4f"%a,"\t\t%.4f"%b,"\t\t%.4f"%c)
    i=i+1

print("______________________________________________________")
print("avg ","\t\t%.4f"%(round(statistics.mean(acc_NB),4)),"\t\t%.4f"%(round(statistics.mean(acc_DT),4)),"\t\t%.4f"%(round(statistics.mean(acc_LR),4)))
print("stdev","\t\t%.4f"%(round(statistics.stdev(acc_NB),4)),"\t\t%.4f"%(round(statistics.stdev(acc_DT),4)),"\t\t%.4f"%(round(statistics.stdev(acc_LR),4)))
print("\nThe average training time is (In seconds)")
print("\t\t",round(statistics.mean(t_NB),3),"\t\t",round(statistics.mean(t_DT),3),"\t\t",round(statistics.mean(t_LR),3))
print("The average F-measure score is")
print("\t\t",(round(statistics.mean(fm_NB),3)),"\t\t",(round(statistics.mean(fm_DT),3)),"\t\t",(round(statistics.mean(fm_LR),3)))

print("============")
print("Table 2")
print("============")
print("____________________________________________________________________")
print("Folds\t\tNaive Bayes\tDecision Tree\tLogistic Regression")
print("____________________________________________________________________")

ranks = [rankdata(row) for row in zip(acc_NB, acc_DT, acc_LR)]
ranks = np.array(ranks)
sum1 = ranks.sum(axis = 0)
avg1 = sum1/10

for i in range(0,10):
    print(i+1,"\t\t%.4f"%acc_NB[i],"(",int(ranks[i][0]),")","\t%.4f"%acc_DT[i],"(",int(ranks[i][1]),")","\t%.4f"%acc_LR[i],"(",int(ranks[i][2]),")")
    
print("______________________________________________________")
print("avg ranks","\t",avg1[0],"\t\t",avg1[1],"\t\t",avg1[2])
    
    