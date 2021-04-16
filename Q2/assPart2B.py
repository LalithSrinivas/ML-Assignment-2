#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 21:24:02 2020

@author: lalithsrinivas
"""

import numpy as np
from cvxopt import solvers, matrix
import pandas as pd
import time


train_file = pd.read_csv('fashion_mnist/train.csv', header=None)
train = []
count = 0
classes = 10
dictionary = {}
for i in range(10):
    for j in range(i+1, 10):
        train.append(train_file.drop(train_file[(train_file[784] != i) & (train_file[784] != j)].index, inplace=False))
        dictionary[i*classes+j] = count
        count+=1
# train_file = None

sol = []
y_final = []
x_final = []
b_star = []
w = []

for l in range(classes):
    for s in range(l+1, classes):
        index = dictionary[l*classes+s]
        print(index, l, s)
        y = np.array(train[index][784])
        y_final.append(y)
        samples = len(train[index])
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
        for p, element in train[index].iterrows():
            x.append((1/255)*np.array(element[:-1]))
            if y[count] == i:
                    y[count] = -1
            if y[count] == j:
                    y[count] = 1
            count+=1
        x_final.append(x)
        for m in range(samples):
            for n in range(m, samples):
                temp = x[m]-x[n]
                temp2 = y[m]*y[n]*np.e**(-0.05*np.dot(temp, temp))
                p1[n][m] = temp2
                p1[m][n] = temp2
            g[m][m] = 1
            g[m+samples][m] = -1
            h[m] = 1
            A[0][m] = y[m]
        
        
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
        sol.append(sol2)
        alpha_guassian = np.array(sol[index]['x'])
        w_pos = np.array([np.inf]*samples)
        w_neg = np.array([-1*np.inf]*samples)
        for i in range(samples):
            if y[i] == -1:
                w_neg[i] = 0
                for j in range(samples):
                    w_neg[i] += alpha_guassian[i]*(p1[i][j]/y[j])
            else:
                w_pos[i] = 0
                for j in range(samples):
                    temp = x[i]-x[j]
                    w_pos[i] += alpha_guassian[i]*y[i]*(np.dot(temp, temp))
        
        min_yg = min(w_pos)
        max_yg = max(w_neg)
        
        b_guassian = -0.5*(max_yg+min_yg)
        b_star.append(b_guassian)
        print(index)
  
test_file = pd.read_csv('fashion_mnist/test.csv', header=None)
count = 0
classes = 10
dictionary = {}

correct_guassian = 0
vote = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
x_train = []
y_train = np.array(test_file[784])
total = len(test_file)
confusion_matrix = [[0]*10]*10
for p, element in test_file.iterrows():
    x_train.append((1/255)*np.array(element[:-1]))
for a in range(total):
    for m in range(10):
        for n in range(10):
            if y_train[i] == m:
                y_train[i] = -1
            if y_train[i] == n:
                y_train = 1
            temp = 0
            index = dictionary[m*10+n]
            samples = len(train[index])
            alpha_guassian = sol[index]['x']
            
            for j in range(samples):
                temp += alpha_guassian[j]*y[index][j]*np.e**(-0.05*np.dot((x_train[a]-x[index][j]), (x_train[a]-x[index][j])))
            
            if temp > 0:
                vote[n] += 1
            else:
                vote[m] += 1
            if y_train[a] == -1:
                y_train[a] = m
            if y_train[a] == 1:
                y_train[a] = n
    max_t = max(vote, key=vote.get)
    if max_t == y[a]:
        correct_guassian += 1
        confusion_matrix[y[a]][y[a]] += 1
    else:
        confusion_matrix[max_t][y[a]] += 1

print("Accuracy ", 100*correct_guassian/total)
print(confusion_matrix)
        
        
        
        
        
        
        
        
        
        
        
        
        