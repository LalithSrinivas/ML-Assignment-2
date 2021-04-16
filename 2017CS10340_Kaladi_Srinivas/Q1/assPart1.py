#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 18:00:56 2020

@author: lalithsrinivas
"""

import csv
import re
import numpy as np
import random
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer(strip_handles=True)
import pandas as pd
from sklearn.naive_bayes import GaussianNB  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.feature_selection import SelectPercentile, chi2
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

regex = re.compile('[^#?a-zA-Z0-9 ]*')
posWordDic = {}
negWordDic = {}
neuWordDic = {}
totalWordDic = {}
numPos = 0
numNeg = 0
numNeu = 0
numPosWords = 0
numNegWords = 0
numNeuWords = 0

with open('trainingandtestdata/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        sentence = regex.sub('', row[-1])
        sentence = sentence.split(' ')
        if row[0] == '0':
            numNeg+=1
            for i in sentence:
                if i != '':
                    numNegWords+= 1
                    if negWordDic.get(i) == None:
                        negWordDic[i] = 1
                    else:
                        negWordDic[i] += 1
                    totalWordDic[i] = 1
        elif row[0] == '4':
            numPos+=1
            for i in sentence:
                if i != '':
                    numPosWords += 1
                    if posWordDic.get(i) == None:
                        posWordDic[i] = 1
                    else:
                        posWordDic[i] += 1
                    totalWordDic[i] = 1

correct = 0
randCorrect=0
majCorrect = 0
total = 0
v =len(totalWordDic)
majority = -1
confusion_matrix = [[0, 0], [0, 0]] #predicted values on rows
if numNeg > numPos:
    majority = 0
else:
    majority = 4
rn = []
prob = []
with open('trainingandtestdata/testdata.manual.2009.06.14.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        if row[0] != '2':
            rn.append(int(row[0]))
            total += 1
            sentence = regex.sub('', row[-1])
            sentence = sentence.split(' ')
            probXY0 = 0
            probXY4 = 0
            for i in sentence:
                if i != '':
                    if posWordDic.get(i) == None:
                        probXY4 += np.log(1/(numPosWords+v))
                    else:
                        probXY4 += np.log((posWordDic.get(i)+1)/(numPosWords+v))
                        
                    if negWordDic.get(i) == None:
                        probXY0 += np.log(1/(numNegWords+v))
                    else:
                        probXY0 += np.log((negWordDic.get(i)+1)/(numNegWords+v))
            y0 = (probXY0+np.log(numNeg))
            y4 = (probXY4+np.log(numPos))
            prob.append(y4/(y0+y4))
            if int(row[0]) == random.choice([0,4]):
                randCorrect += 1
            if int(row[0]) == majority:
                majCorrect += 1
            if y0 > y4:
                if int(row[0]) == 0:
                    correct+=1
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1
            else:
                if int(row[0]) == 4:
                    correct+=1
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[1][0] += 1
a, b, c = roc_curve(rn, prob, pos_label=4)
plt.plot(b, a)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()
print ("For sample, accuracy is:", 100*correct/total)
print ("For sample, random prediction accuracy is:", 100*randCorrect/total)
print ("For sample, majority prediction accuracy is:", 100*majCorrect/total)
print("Confusion Matrix: ", confusion_matrix)
      
stop_words = stopwords.words('english')  
with open('trainingandtestdata/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        sentence = regex.sub('', row[-1])
        init_sentence = sentence.split(' ')
        sentence = [i for i in init_sentence if i not in stop_words]
        if row[0] == '0':
            numNeg+=1
            for i in sentence:
                if i != '':
                    numNegWords+= 1
                    if negWordDic.get(i) == None:
                        negWordDic[i] = 1
                    else:
                        negWordDic[i] += 1
                    totalWordDic[i] = 1
        elif row[0] == '4':
            numPos+=1
            for i in sentence:
                if i != '':
                    numPosWords += 1
                    if posWordDic.get(i) == None:
                        posWordDic[i] = 1
                    else:
                        posWordDic[i] += 1
                    totalWordDic[i] = 1
                    
                    
                    
correct = 0
randCorrect=0
majCorrect = 0
total = 0
v =len(totalWordDic)
majority = -1
confusion_matrix = [[0, 0], [0, 0]] #predicted values on rows
if numNeg > numPos:
    majority = 0
else:
    majority = 4
with open('trainingandtestdata/testdata.manual.2009.06.14.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        if row[0] != '2':
            total += 1
            sentence = regex.sub('', row[-1])
            init_sentence = sentence.split(' ')
            sentence = [i for i in init_sentence if i not in stop_words]
            probXY0 = 0
            probXY4 = 0
            for i in sentence:
                if i != '':
                    if posWordDic.get(i) == None:
                        probXY4 += np.log(1/(numPosWords+v))
                    else:
                        probXY4 += np.log((posWordDic.get(i)+1)/(numPosWords+v))
                        
                    if negWordDic.get(i) == None:
                        probXY0 += np.log(1/(numNegWords+v))
                    else:
                        probXY0 += np.log((negWordDic.get(i)+1)/(numNegWords+v))
            y0 = (probXY0+np.log(numNeg))
            y4 = (probXY4+np.log(numPos))
            if int(row[0]) == random.choice([0,4]):
                randCorrect += 1
            if int(row[0]) == majority:
                majCorrect += 1
            if y0 > y4:
                if int(row[0]) == 0:
                    correct+=1
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1
            else:
                if int(row[0]) == 4:
                    correct+=1
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[1][0] += 1
print("after removing stop words")
print ("For sample, accuracy is:", 100*correct/total)
print ("For sample, random prediction accuracy is:", 100*randCorrect/total)
print ("For sample, majority prediction accuracy is:", 100*majCorrect/total)
print("Confusion Matrix: ", confusion_matrix)

posWordDic = {}
negWordDic = {}
neuWordDic = {}
totalWordDic = {}
numPos = 0
numNeg = 0
numNeu = 0
numPosWords = 0
numNegWords = 0
numNeuWords = 0
stop_words = stopwords.words('english')  
ps = PorterStemmer()
with open('trainingandtestdata/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        init_sentence = tknzr.tokenize(row[-1])
        init_sentence = [i for i in init_sentence if i not in stop_words]
        sentence = [ps.stem(i) for i in init_sentence]
        if row[0] == '0':
            numNeg+=1
            for i in sentence:
                if i != '':
                    numNegWords+= 1
                    if negWordDic.get(i) == None:
                        negWordDic[i] = 1
                    else:
                        negWordDic[i] += 1
                    totalWordDic[i] = 1
        elif row[0] == '4':
            numPos+=1
            for i in sentence:
                if i != '':
                    numPosWords += 1
                    if posWordDic.get(i) == None:
                        posWordDic[i] = 1
                    else:
                        posWordDic[i] += 1
                    totalWordDic[i] = 1
                    
                    
                    
correct = 0
randCorrect=0
majCorrect = 0
total = 0
v =len(totalWordDic)
majority = -1
confusion_matrix = [[0, 0], [0, 0]] #predicted values on rows
if numNeg > numPos:
    majority = 0
else:
    majority = 4
with open('trainingandtestdata/testdata.manual.2009.06.14.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        if row[0] != '2':
            total += 1
            sentence = regex.sub('', row[-1])
            init_sentence = sentence.split(' ')
            init_sentence = [i for i in init_sentence if i not in stop_words]
            sentence = [ps.stem(i) for i in init_sentence]
            probXY0 = 0
            probXY4 = 0
            for i in sentence:
                if i != '':
                    if posWordDic.get(i) == None:
                        probXY4 += np.log(1/(numPosWords+v))
                    else:
                        probXY4 += np.log((posWordDic.get(i)+1)/(numPosWords+v))
                        
                    if negWordDic.get(i) == None:
                        probXY0 += np.log(1/(numNegWords+v))
                    else:
                        probXY0 += np.log((negWordDic.get(i)+1)/(numNegWords+v))
            y0 = (probXY0+np.log(numNeg))
            y4 = (probXY4+np.log(numPos))
            if int(row[0]) == random.choice([0,4]):
                randCorrect += 1
            if int(row[0]) == majority:
                majCorrect += 1
            if y0 > y4:
                if int(row[0]) == 0:
                    correct+=1
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1
            else:
                if int(row[0]) == 4:
                    correct+=1
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[1][0] += 1
print("after removing stop words, username tokens")
print ("For sample, accuracy is:", 100*correct/total)
print ("For sample, random prediction accuracy is:", 100*randCorrect/total)
print ("For sample, majority prediction accuracy is:", 100*majCorrect/total)
print("Confusion Matrix: ", confusion_matrix)

g=GaussianNB()
count = 0
training_chunks = pd.read_csv('trainingandtestdata/training.1600000.processed.noemoticon.csv', header=None, encoding='latin-1')
voca = list(totalWordDic.keys())
v = TfidfVectorizer(vocabulary=voca, use_idf = True, dtype=np.float32)
x = training_chunks[5]
y = training_chunks[0]
sp = SelectPercentile(chi2, percentile=0.1)
sp.fit_transform(v.fit_transform(x.astype('U')), y)
x = sp.get_support()
voc = []


for i in range(len(voca)):
    if x[i]:
        voc.append(voca[i])

v = TfidfVectorizer(vocabulary=voc, use_idf = True, dtype=np.float32)
for i in range(0, 1600000, 10000):
    # print(training_df)
    x = training_chunks[5][i: i+10000]
    y = training_chunks[0][i: i+10000]

    x = v.fit_transform(x.astype('U')).toarray()
    g = g.partial_fit(x, y, classes=[0, 4])
    print(count)
    count+=1
correct = 0
# randCorrect=0
# majCorrect = 0
total = 0
# v =len(totalWordDic)
majority = -1
confusion_matrix = [[0, 0], [0, 0]] #predicted values on rows
# if numNeg > numPos:
#     majority = 0
# else:
#     majority = 4
temp = pd.read_csv('trainingandtestdata/testdata.manual.2009.06.14.csv', encoding='latin-1', header=None)
x = temp[5]
y= temp[0]
x = v.transform(x.astype('U')).toarray()
result = g.predict(x)
for i in range(len(result)):
    if y[i] != '2':
        total+=1
        if result[i] == int(y[i]):
            correct+=1
print ("For sample, accuracy is:", 100*correct/total)
# print("Confusion Matrix: ", confusion_matrix)

posWordDic = {}
negWordDic = {}
neuWordDic = {}
totalWordDic = {}
numPos = 0
numNeg = 0
numNeu = 0
numPosWords = 0
numNegWords = 0
numNeuWords = 0
stop_words = stopwords.words('english')  
ps = PorterStemmer()
with open('trainingandtestdata/training.1600000.processed.noemoticon.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        init_sentence = tknzr.tokenize(row[-1])
        init_sentence = [i for i in init_sentence if i not in stop_words]
        sentence = [ps.stem(i) for i in init_sentence]
        sentence = sentence + list(nltk.bigrams(sentence))
        if row[0] == '0':
            numNeg+=1
            for i in sentence:
                if i != '':
                    numNegWords+= 1 
                    if negWordDic.get(i) == None:
                        negWordDic[i] = 1
                    else:
                        negWordDic[i] += 1
                    totalWordDic[i] = 1
        elif row[0] == '4':
            numPos+=1
            for i in sentence:
                if i != '':
                    numPosWords += 1
                    if posWordDic.get(i) == None:
                        posWordDic[i] = 1
                    else:
                        posWordDic[i] += 1
                    totalWordDic[i] = 1
correct = 0
randCorrect=0
majCorrect = 0
total = 0
v =len(totalWordDic)
majority = -1
confusion_matrix = [[0, 0], [0, 0]] #predicted values on rows
if numNeg > numPos:
    majority = 0
else:
    majority = 4
with open('trainingandtestdata/testdata.manual.2009.06.14.csv', newline='', encoding='latin-1') as file:
    rows = csv.reader(file, delimiter=',')
    for row in rows:
        if row[0] != '2':
            total += 1
            sentence = regex.sub('', row[-1])
            init_sentence = sentence.split(' ')
            init_sentence = [i for i in init_sentence if i not in stop_words]
            sentence = [ps.stem(i) for i in init_sentence]
            sentence += list(nltk.bigrams(sentence))
            probXY0 = 0
            probXY4 = 0
            for i in sentence:
                if i != '':
                    if posWordDic.get(i) == None:
                        probXY4 += np.log(1/(numPosWords+v))
                    else:
                        probXY4 += np.log((posWordDic.get(i)+1)/(numPosWords+v))
                        
                    if negWordDic.get(i) == None:
                        probXY0 += np.log(1/(numNegWords+v))
                    else:
                        probXY0 += np.log((negWordDic.get(i)+1)/(numNegWords+v))
            y0 = (probXY0+np.log(numNeg))
            y4 = (probXY4+np.log(numPos))
            if int(row[0]) == random.choice([0,4]):
                randCorrect += 1
            if int(row[0]) == majority:
                majCorrect += 1
            if y0 > y4:
                if int(row[0]) == 0:
                    correct+=1
                    confusion_matrix[0][0] += 1
                else:
                    confusion_matrix[0][1] += 1
            else:
                if int(row[0]) == 4:
                    correct+=1
                    confusion_matrix[1][1] += 1
                else:
                    confusion_matrix[1][0] += 1
print("after removing stop words, username tokens")
print ("For sample, accuracy is:", 100*correct/total)
print ("For sample, random prediction accuracy is:", 100*randCorrect/total)
print ("For sample, majority prediction accuracy is:", 100*majCorrect/total)
print("Confusion Matrix: ", confusion_matrix)