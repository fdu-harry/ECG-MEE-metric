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

def baseline_determine(ecg,band_num):
    ecg_upper=np.max(ecg)
    ecg_lower=np.min(ecg)
    sample_count=np.zeros(band_num)
    for i in range(len(ecg)):
        if ecg[i]==ecg_upper:
            sample_count[band_num-1]+=1
        if ecg[i]!=ecg_upper:
            sample_count[int(np.floor((ecg[i]-ecg_lower)*band_num))]+=1
    baseline=np.argmax(sample_count)
    if baseline!=0:
        baseline_lower=baseline*(1/band_num)
        baseline_upper=(baseline+1)*(1/band_num)
    if baseline==0:
        sample_count.sort()
        baseline=sample_count[-2]
        baseline_lower=baseline*(1/band_num)
        baseline_upper=(baseline+1)*(1/band_num)
    return baseline_upper,baseline_lower

def diff(data):
    data=data.flatten()
    temp=np.zeros(len(data))
    temp[1:]=data[1:]-data[:-1]
    return temp

def equal_position(data):
    equal_position=0
    temp=np.sum(data)/2
    for i in range(len(data)):
        if np.sum(data[:i])<temp and np.sum(data[:i+1])>temp:
            equal_position=i
    return equal_position

def wave_set_determine(ecg,band_num):
    upper_wave_set=[]
    lower_wave_set=[]
    baseline_upper,baseline_lower=baseline_determine(ecg,band_num)
    baseline=(baseline_upper+baseline_lower)/2
    upper_wave=np.argwhere(ecg>baseline_upper)
    lower_wave=np.argwhere(ecg<baseline_lower)
    upper_wave_diff=diff(upper_wave)
    lower_wave_diff=diff(lower_wave)
    upper_wave_index=np.argwhere(upper_wave_diff>1)
    lower_wave_index=np.argwhere(lower_wave_diff>1)
    #upper_wave
    if len(upper_wave_index)==0:
        upper_wave_set.append(upper_wave[:len(upper_wave_diff)-1])
    if len(upper_wave_index)!=0:
        if upper_wave_index[0][0]>2:
            upper_wave_set.append(upper_wave[:upper_wave_index[0][0]])
        for i in range(0,len(upper_wave_index)-1):
            if upper_wave_index[i+1]-upper_wave_index[i]>2:
                upper_wave_set.append(upper_wave[upper_wave_index[i][0]:upper_wave_index[i+1][0]])
        if len(upper_wave)-(upper_wave_index[-1][0])>2:
            upper_wave_set.append(upper_wave[upper_wave_index[-1][0]+1:])
    #lower_wave
    if len(lower_wave_index)==0:
        lower_wave_set.append(lower_wave[:len(lower_wave_diff)-1])
    if len(lower_wave_index)!=0:
        if lower_wave_index[0][0]>2:
            lower_wave_set.append(lower_wave[:lower_wave_index[0][0]])
        for i in range(0,len(lower_wave_index)-1):
            if lower_wave_index[i+1]-lower_wave_index[i]>2:
                lower_wave_set.append(lower_wave[lower_wave_index[i][0]:lower_wave_index[i+1][0]])
        if len(lower_wave)-(lower_wave_index[-1][0])>2:
            lower_wave_set.append(lower_wave[lower_wave_index[-1][0]:])
    return upper_wave_set,lower_wave_set,baseline

def BWE(ecg,band_num):
    baseline_upper,baseline_lower = baseline_determine(ecg,band_num)
    baseline_mid = (baseline_upper+baseline_lower)/2
    count = np.zeros(band_num)
    for i in range(len(ecg)):
        if ecg[i]==1:
            count[band_num-1]+=1
        if ecg[i]!=1:
            count[int(np.floor(ecg[i]/(1/band_num)))]+=1
    count_fz = count/np.sum(count)
    band_mid = np.arange(0,1,1/band_num)+1/2*(1/band_num)
    band_entropy = -np.sum(band_mid*np.nan_to_num(np.log2(count_fz)*count_fz))
    return band_entropy,baseline_mid

def WSE(ecg,band_num):
    ecg_entropy=0
    upper_wave_set,lower_wave_set,baseline=wave_set_determine(ecg,band_num)
    for i in range(len(upper_wave_set)):
        wave=ecg[upper_wave_set[i]]-baseline
        average_voltage=np.sum(wave)/len(wave)
        wave_fz=len(wave)/len(ecg)
        wave_entropy = -np.log2(wave_fz)*wave_fz*average_voltage+equal_position(wave)*10/len(ecg)*average_voltage
        ecg_entropy+=wave_entropy
    for i in range(len(lower_wave_set)):
        wave=-(ecg[lower_wave_set[i]]-baseline)
        average_voltage=np.sum(wave)/len(wave)
        wave_fz=len(wave)/len(ecg)
        wave_entropy = -np.log2(wave_fz)*wave_fz*average_voltage+equal_position(wave)*10/len(ecg)*average_voltage
        ecg_entropy+=wave_entropy
    return ecg_entropy

def MEE(bwe,wse):
    mee1 = bwe*math.exp(wse/(2*math.pi))
    mee2 = (bwe**2+wse**2)/2
    return mee1, mee2

def ecg_plot(ecg,band_num):
    upper_wave_set,lower_wave_set,baseline=wave_set_determine(ecg,band_num)
    baseline_upper,baseline_lower=baseline_determine(ecg,band_num)
    plt.figure(figsize=(14,3))
    plt.subplot(1,2,1)
    plt.plot(ecg)
    plt.subplot(1,2,2)
    for i in range(len(upper_wave_set)):
        x=upper_wave_set[i]
        plt.plot(x,ecg[upper_wave_set[i]])
    for i in range(len(lower_wave_set)):
        x=lower_wave_set[i]
        plt.plot(x,ecg[lower_wave_set[i]])
    plt.axhline(y=baseline_upper,color='blueviolet')
    plt.axhline(y=baseline_lower,color='blueviolet')
    plt.show()