#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:30:49 2020

@author: lalithsrinivas
"""

import numpy as np
from cvxopt import solvers, matrix
import pandas as pd
import time
train_file = pd.read_csv('fashion_mnist/train.csv', header=None)
train_file.drop(train_file[train_file[784] > 1].index, inplace=True)
y = np.array(train_file[784])
samples = len(train_file)
g = np.zeros((2*samples, samples))
h = np.zeros((2*samples, 1))
A = np.zeros((1, samples))
b = 0
q = -1+np.zeros((samples, 1))
p1 = np.zeros((samples, samples))
p2 = np.zeros((samples, samples))
x = []
b_gaussian = 0
count=0
start_time = time.time()
for index, element in train_file.iterrows():
    x.append((1/255)*np.array(element[:-1]))
    if y[count] == 0:
            y[count] = -1
    count+=1
    
for i in range(samples):
    for j in range(i, samples):
        temp = x[i]-x[j]
        temp2 = y[i]*y[j]*np.e**(-0.05*np.dot(temp, temp))
        temp3 = y[i]*y[j]*np.dot(x[i], x[j])
        p1[i][j] = temp2
        p1[j][i] = temp2
        p2[i][j] = temp3
        p2[j][i] = temp3
    g[i][i] = 1
    g[i+samples][i] = -1
    h[i] = 1
    A[0][i] = y[i]


# for i in range(samples):
#     for j in range(samples):
#         p2[i][j] = y[i]*y[j]*np.dot(x[i], x[j])
        
g = matrix(g, tc='d')
h = matrix(h, tc='d')
A = matrix(A, tc='d')
b = matrix([b], tc='d')
q = matrix(q, tc='d')
P = matrix(p1, tc='d')

sol2 = solvers.qp(P, q, g, h, A, b)

print("guassian execution time", time.time()-start_time)
start_time = time.time()

P = matrix(p2, tc='d')

sol1 = solvers.qp(P, q, g, h, A, b)
print("linear execution time", time.time()-start_time)
start_time = time.time()

alpha_linear = np.array(sol1['x'])
alpha_guassian = np.array(sol2['x'])
w_star = np.array([0]*784, dtype=np.float64)
b_star = 0
min_y = np.inf
max_y = -1*np.inf
w_pos = np.array([np.inf]*samples)
w_neg = np.array([-1*np.inf]*samples)
for i in range(samples):
    w_star += alpha_linear[i]*y[i]*x[i]
    if y[i] == -1:
        w_neg[i] = 0
        for j in range(samples):
            w_neg[i] += alpha_guassian[i]*(p1[i][j]/(y[j]))
    else:
        w_pos[i] = 0
        for j in range(samples):
            w_pos[i] += alpha_guassian[i]*(p1[i][j]/(y[j]))
for i in range(samples):
    if y[i] == 1:
        min_y = min(np.dot(w_star, x[i]), min_y)
    else:
        max_y = max(np.dot(w_star, x[i]), max_y)

min_yg = min(w_pos)
max_yg = max(w_neg)
b_star = -0.5*(max_y + min_y)
b_guassian = -0.5*(max_yg+min_yg)

test_file = pd.read_csv('fashion_mnist/test.csv', header = None)
test_file.drop(test_file[test_file[784] > 1].index, inplace=True)
y_train = np.array(test_file[784])
x_train = []
total = len(test_file)
correct =0
correct_guassian = 0
for index_train, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
correct =0
correct_guassian = 0
for i in range(total):
    if y_train[i] == 0:
        y_train[i] = -1
    temp = 0
    for j in range(samples):
        temp += alpha_guassian[j]*y[j]*np.e**(-0.05*np.dot((x_train[i]-x[j]), (x_train[i]-x[j])))
    if ((np.dot(w_star, x_train[i])+b_star)*y_train[i] > 0):
        correct += 1
    if ((temp+b_guassian)*y_train[i] > 0):
        correct_guassian += 1
print("linear SVM accuracy ", 100*(correct/total))
print("guassian SVM accuracy ", 100*(correct_guassian/total))
print("test execution time", time.time()-start_time)

from sklearn.svm import SVC

clf = SVC(kernel = 'rbf', gamma=0.05)

clf.fit(np.array(x), y)

print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))

clf = SVC(kernel = 'linear')

clf.fit(np.array(x), y)

print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))# def kernel(x_train, z):
#     return np.float64(np.e**(-0.05*np.dot((x_train-z), (x_train-z))))

print("sklearn execution time", time.time()-start_time)
start_time = time.time()

test_file = pd.read_csv('fashion_mnist/val.csv', header = None)
test_file.drop(test_file[test_file[784] > 1].index, inplace=True)
y_train = np.array(test_file[784])
x_train = []
total = len(test_file)
correct =0
correct_guassian = 0
for index_train, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
correct =0
correct_guassian = 0
for i in range(total):
    if y_train[i] == 0:
        y_train[i] = -1
    temp = 0
    for j in range(samples):
        temp += alpha_guassian[j]*y[j]*np.e**(-0.05*np.dot((x_train[i]-x[j]), (x_train[i]-x[j])))
    if ((np.dot(w_star, x_train[i])+b_star)*y_train[i] > 0):
        correct += 1
    if ((temp+b_guassian)*y_train[i] > 0):
        correct_guassian += 1
print("linear SVM accuracy ", 100*(correct/total))
print("guassian SVM accuracy ", 100*(correct_guassian/total))
print("test execution time", time.time()-start_time)
start_time = time.time()

clf = SVC(kernel = 'rbf', gamma=0.05)

clf.fit(np.array(x), y)

print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))

clf = SVC(kernel = 'linear')

clf.fit(np.array(x), y)

print("sklearn accuracy", 100*clf.score(np.array(x_train), y_train))# def kernel(x_train, z):
#     return np.float64(np.e**(-0.05*np.dot((x_train-z), (x_train-z))))

print("sklearn execution time", time.time()-start_time)
start_time = time.time()