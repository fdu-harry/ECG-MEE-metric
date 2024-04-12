# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:02:03 2024

@author: DrHu
"""
import itertools as it
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import math

def ApEn(x, m, r):                                             
    N = len(x)                                                     
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("Not one-dimension")
    if N < m + 1:
        raise ValueError("len(x)<m+1")  

    phi = []
    for temp in range(2):  
       
        X = []
        m = m+temp
        for i in range(N + 1 - m):
            X.append(x[i:i+m])
        X = np.array(X)
        
        C = []
        for index1, i in enumerate(X):
            count = 0
            for index2, j in enumerate(X):
                if index1 != index2:
                    if np.max(np.abs(i-j)) <= r:
                        count += 1
            
            C.append(count/(N-m+1))
        
        C = np.array(C)
        C = np.where(C == 0, 1e-10, C)

        phi.append(np.sum(np.log(C))/(N-m+1))
        
    apEn = phi[0] - phi[1]
    return apEn

def SampEn(x, m, r):
    N = len(x)                                                     
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("Not one-dimension")
    if N < m + 1:
        raise ValueError("len(x)<m+1")  
        
    AB = []
    for temp in range(2):  
        X = []
        m = m+temp
        for i in range(N + 1 - m):
            X.append(x[i:i+m])
        X = np.array(X)
        
        C = []
        for index1, i in enumerate(X):
            count = 0
            for index2, j in enumerate(X):
                if index1 != index2:
                    if np.max(np.abs(i-j)) <= r:
                        count += 1
            C.append(count/(N-m+1))
        C = np.array(C)
        C = np.where(C == 0, 1e-10, C)
        
        AB.append(np.sum(C)/(N-m+1))
        
    SE = np.log(AB[0]) - np.log(AB[1])
    return SE

def FuzzyEntopy(x, m, r, n=2):
    x = np.array(x)
    if x.ndim != 1:
        raise ValueError("Not one-dimension")
    if len(x) < m + 1:
        raise ValueError("len(x)<m+1")  
    entropy = 0
    
    for temp in range(2):  
        X = []
        for i in range(len(x) + 1 - (m  + temp)):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        D_value = []
        for index1, i in enumerate(X):
            d = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    d.append(np.max(np.abs(i-j)))
            D_value.append(d)
        D = np.exp(-np.power(D_value, n)/r)
        Lm = np.average(D.ravel())
        entropy = np.abs(entropy) - np.log(Lm)
    return entropy
    
def PE(y, D, t):
    y_len = len(y)
    serial = np.arange(0, D)
    y_perm = list(it.permutations(serial, D))
    count = np.zeros(len(y_perm))
    '''for item in y_perm:
        print(item)
    '''

    for i in range(y_len-(D-1)*t):
        y_x = np.argsort(y[i:i+t*D:t])

        for j in range(len(y_perm)):
            if tuple(y_x) == y_perm[j]:
                count[j] += 1

    pe = scipy.stats.entropy(count / (y_len-(D-1)*t), base=2)
    nor_pe = pe/math.log(math.factorial(D),2)
    return nor_pe

'''
D = 3, t = 1
Partition: [4, 7, 9, 10, 6, 11, 3]
 
 [ 4,  7,  9]->[0, 1, 2]
 [ 7,  9, 10]->[0, 1, 2]
 [ 9, 10,  6]->[1, 2, 0]
 [10,  6, 11]->[1, 0, 2]
 [ 6, 11,  3]->[1, 2, 0]
 
 pe = -∑pi*log(pi) = -(2/5*log2(2/5) + 1/5*log2(1/5) + 2/5*log2(2/5)) = 1.5219
'''

# *******************************************************************************
def BWE():
Upon resolution of arXiv-on-hold, immediate disclosure will be made.

def WSE():
Upon resolution of arXiv-on-hold, immediate disclosure will be made.

def MEE(bwe,wse):
    mee1 = bwe*math.exp(wse/(2*math.pi))
    mee2 = (bwe**2+wse**2)/2
    return mee1, mee2
