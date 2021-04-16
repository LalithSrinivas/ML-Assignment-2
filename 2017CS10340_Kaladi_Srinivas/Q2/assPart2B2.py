#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 13:38:29 2020

@author: lalithsrinivas
"""

from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
import pandas as pd

import numpy as np

from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt

train_file = pd.read_csv('fashion_mnist/train.csv', header=None)

y = np.array(train_file[784])
x = []
for index, element in train_file.iterrows():
    x.append((1/255)*np.array(element[:-1]))

clf = SVC(kernel = 'rbf', gamma=0.05, decision_function_shape = 'ovo')

clf.fit(x, y)

test_file = pd.read_csv('fashion_mnist/test.csv', header = None)

y_train = np.array(test_file[784])

x_train = []

for index_train, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
    
print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))

disp = plot_confusion_matrix(clf, np.array(x_train), y_train, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

test_file = pd.read_csv('fashion_mnist/val.csv', header = None)

y_train = np.array(test_file[784])

x_train = []

for index_train, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
    
print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))

disp = plot_confusion_matrix(clf, np.array(x_train), y_train, display_labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
C = [10**(-5), 10**(-3), 1, 5, 10]
for c in C:
    clf = SVC(kernel = 'rbf', gamma=0.05, C=c, decision_function_shape = 'ovo')
    
    scores = cross_validate(clf, x, y, cv=5)
    
    print('test_scores with C =', c, 'is ', scores['test_scores'])
    
train_file = pd.read_csv('fashion_mnist/train.csv', header=None)

y = np.array(train_file[784])
x = []
for index, element in train_file.iterrows():
    x.append((1/255)*np.array(element[:-1]))
test_file = pd.read_csv('fashion_mnist/test.csv', header = None)

y_train = np.array(test_file[784])

x_train = []
for index_train, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
for c in C:
    clf = SVC(kernel = 'rbf', gamma=0.05, C=c, decision_function_shape = 'ovo')
    clf.fit(x, y)
    print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))