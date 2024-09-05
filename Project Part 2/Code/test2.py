#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft,rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from joblib import dump, load

data=pd.read_csv('test.csv',header=None)

def nomealmatrix_creation(dta_nomeal):
    idxrem_nonmeal=dta_nomeal.isna().sum(axis=1).replace(0,np.nan).dropna().where(lambda x:x>5).dropna().index
    nonmeal_dtcle=dta_nomeal.drop(dta_nomeal.index[idxrem_nonmeal]).reset_index().drop(columns='index')
    nonmeal_dtcle=nonmeal_dtcle.interpolate(method='linear',axis=1)
    idxdpag=nonmeal_dtcle.isna().sum(axis=1).replace(0,np.nan).dropna().index
    nonmeal_dtcle=nonmeal_dtcle.drop(nonmeal_dtcle.index[idxdpag]).reset_index().drop(columns='index')
    ftxmat_nomeal=pd.DataFrame()
    power_first_max=[]
    index_first_max=[]
    power_second_max=[]
    index_second_max=[]
    power_third_max=[]
    for i in range(len(nonmeal_dtcle)):
        array=abs(rfft(nonmeal_dtcle.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array=abs(rfft(nonmeal_dtcle.iloc[:,0:24].iloc[i].values.tolist())).tolist()
        sorted_array.sort()
        power_first_max.append(sorted_array[-2])
        power_second_max.append(sorted_array[-3])
        power_third_max.append(sorted_array[-4])
        index_first_max.append(array.index(sorted_array[-2]))
        index_second_max.append(array.index(sorted_array[-3]))
    ftxmat_nomeal['power_second_max']=power_second_max
    ftxmat_nomeal['power_third_max']=power_third_max
    first_differential_data=[]
    second_differential_data=[]
    standard_deviation=[]
    for i in range(len(nonmeal_dtcle)):
        first_differential_data.append(np.diff(nonmeal_dtcle.iloc[:,0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(nonmeal_dtcle.iloc[:,0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(nonmeal_dtcle.iloc[i]))
    ftxmat_nomeal['2ndDifferential']=second_differential_data
    ftxmat_nomeal['standard_deviation']=standard_deviation
    return ftxmat_nomeal

data1=nomealmatrix_creation(data)

from joblib import dump, load
with open('RandomForestClassifier.pickle', 'rb') as pr_train:
    pic_fi = load(pr_train)
    pre = pic_fi.predict(data1)
    pr_train.close()

pd.DataFrame(pre).to_csv('Result.csv',index=False,header=False)