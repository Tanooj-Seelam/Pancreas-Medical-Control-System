#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from joblib import dump, load

data_df = pd.read_csv('test.csv', header=None)


def nomeal_matcrt(dta_nomeal):
    idx_nonmeal = dta_nomeal.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 5).dropna().index
    no_meal_dt = dta_nomeal.drop(dta_nomeal.index[idx_nonmeal]).reset_index().drop(columns='index')
    no_meal_dt = no_meal_dt.interpolate(method='linear', axis=1)
    id_pag = no_meal_dt.isna().sum(axis=1).replace(0, np.nan).dropna().index
    no_meal_dt = no_meal_dt.drop(no_meal_dt.index[id_pag]).reset_index().drop(columns='index')
    ftmatx_nomeal = pd.DataFrame()
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i in range(len(no_meal_dt)):
        ary = abs(rfft(no_meal_dt.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sor_ary = abs(rfft(no_meal_dt.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        sor_ary.sort()
        power_first_max.append(sor_ary[-2])
        power_second_max.append(sor_ary[-3])
        power_third_max.append(sor_ary[-4])
        index_first_max.append(ary.index(sor_ary[-2]))
        index_second_max.append(ary.index(sor_ary[-3]))
    ftmatx_nomeal['power_second_max'] = power_second_max
    ftmatx_nomeal['power_third_max'] = power_third_max
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(no_meal_dt)):
        first_differential_data.append(np.diff(no_meal_dt.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(no_meal_dt.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(no_meal_dt.iloc[i]))
    ftmatx_nomeal['2ndDifferential'] = second_differential_data
    ftmatx_nomeal['standard_deviation'] = standard_deviation
    return ftmatx_nomeal


data1 = nomeal_matcrt(data_df)

from joblib import dump, load

with open('RandomForestClassifier.pickle', 'rb') as pr_train:
    pc_ft = load(pr_train)
    predict = pc_ft.predict(data1)
    pr_train.close()

pd.DataFrame(predict).to_csv('Result.csv', index=False, header=False)
