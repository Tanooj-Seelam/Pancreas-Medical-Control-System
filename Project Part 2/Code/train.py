#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.fftpack import fft, ifft, rfft
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold, train_test_split, cross_val_score
from joblib import dump, load

insline_df = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

cgdf_df = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

insline_df['date_time_stamp'] = pd.to_datetime(insline_df['Date'] + ' ' + insline_df['Time'])
cgdf_df['date_time_stamp'] = pd.to_datetime(cgdf_df['Date'] + ' ' + cgdf_df['Time'])

inslinealt_df = pd.read_csv('Insulin_patient2.csv', low_memory=False,usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

cgdfalt_df = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

inslinealt_df['date_time_stamp'] = pd.to_datetime(inslinealt_df['Date'] + ' ' + inslinealt_df['Time'])
cgdfalt_df['date_time_stamp'] = pd.to_datetime(cgdfalt_df['Date'] + ' ' + cgdfalt_df['Time'])


def mealdata_creation(insline_df, cgdf_df, date_identi):
    instance_df = insline_df.copy()
    instance_df = instance_df.set_index('date_time_stamp')

    valid_tsdf = instance_df.sort_values(by='date_time_stamp', ascending=True).dropna().reset_index()

    valid_tsdf['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    valid_tsdf = valid_tsdf.dropna().reset_index().drop(columns='index')

    tsp_list = []
    bho = 0
    for ch , it in enumerate(valid_tsdf['date_time_stamp']):
        try:
            bho = (valid_tsdf['date_time_stamp'][ch + 1] - it).seconds / 60.0
            if bho >= 120:
                tsp_list.append(it)
        except KeyError:
            break

    mlgluc_lst = []

    ch = 0
    while ch < len(tsp_list):
        ie = tsp_list[ch]
        st = pd.to_datetime(ie - timedelta(minutes=30))
        en = pd.to_datetime(ie + timedelta(minutes=120))
        if date_identi == 1:
            gtdta = ie.date().strftime('%m/%d/%Y')
        else:
            gtdta = ie.date().strftime('%Y-%m-%d')
        gluc_val = cgdf_df.loc[cgdf_df['Date'] == gtdta].set_index('date_time_stamp').between_time(
            start_time=st.strftime('%H:%M:%S'), end_time=en.strftime('%H:%M:%S'))[
            'Sensor Glucose (mg/dL)'].values.tolist()
        mlgluc_lst.append(gluc_val)
        ch += 1

    return pd.DataFrame(mlgluc_lst)


def mealdata_cret(inslie_df, cmg_df, num_gr):
    mealdta_df = mealdata_creation(inslie_df, cmg_df, num_gr)
    print(mealdta_df.iloc[:, 0:30])
    return mealdta_df.iloc[:, 0:30]


num_list = [1, 2]
for i in num_list:
    if i == num_list[0]:
        mealdta_df = mealdata_cret(insline_df, cgdf_df, i)
    else:
        exmealdta = mealdata_cret(inslinealt_df, cgdfalt_df, i)


### No-Meal data extraction

def nomeal_dtacreat(insline_df, cgdf_df):
    insu_nomeal = insline_df.copy()
    tet1_df = insu_nomeal.sort_values(by='date_time_stamp', ascending=True).replace(0.0, np.nan).dropna().copy()
    tet1_df = tet1_df.reset_index().drop(columns='index')
    vld_tmp = []
    for ix, f in enumerate(tet1_df['date_time_stamp']):
        try:
            vlu = (tet1_df['date_time_stamp'][ix + 1] - f).seconds // 3600
            if vlu >= 4:
                vld_tmp.append(f)
        except KeyError:
            break
    dt_set = []
    for ifx, ir in enumerate(vld_tmp):
        itr_ds = 1
        try:
            lth_dataset24 = len(cgdf_df.loc[(cgdf_df['date_time_stamp'] >= vld_tmp[
                ifx] + pd.Timedelta(hours=2)) & (cgdf_df['date_time_stamp'] < vld_tmp[ifx + 1])]) // 24
            while (itr_ds <= lth_dataset24):
                if itr_ds == 1:
                    dt_set.append(cgdf_df.loc[(cgdf_df['date_time_stamp'] >= vld_tmp[
                        ifx] + pd.Timedelta(hours=2)) & (cgdf_df['date_time_stamp'] < vld_tmp[ifx + 1])][
                                       'Sensor Glucose (mg/dL)'][:itr_ds * 24].values.tolist())
                    itr_ds += 1
                else:
                    dt_set.append(cgdf_df.loc[(cgdf_df['date_time_stamp'] >= vld_tmp[
                        ifx] + pd.Timedelta(hours=2)) & (cgdf_df['date_time_stamp'] < vld_tmp[ifx + 1])][
                                       'Sensor Glucose (mg/dL)'][
                                   (itr_ds - 1) * 24:(itr_ds) * 24].values.tolist())
                    itr_ds += 1
        except IndexError:
            break
    return pd.DataFrame(dt_set)


dt_nomeal = nomeal_dtacreat(insline_df, cgdf_df)
exdt_nomeal = nomeal_dtacreat(inslinealt_df, cgdfalt_df)


def matrix_mlcreat(input_mealdata):
    itx = input_mealdata.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    cleandata_inp = input_mealdata.drop(input_mealdata.index[itx]).reset_index().drop(columns='index')
    cleandata_inp = cleandata_inp.interpolate(method='linear', axis=1)
    indrop_ag = cleandata_inp.isna().sum(axis=1).replace(0, np.nan).dropna().index
    cleandata_inp = cleandata_inp.drop(input_mealdata.index[indrop_ag]).reset_index().drop(columns='index')
    cleandata_inp = cleandata_inp.dropna().reset_index().drop(columns='index')
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for it, rowdt in enumerate(cleandata_inp.iloc[:, 0:30].values.tolist()):
        ara = abs(rfft(rowdt)).tolist()
        sort_ara = abs(rfft(rowdt)).tolist()
        sort_ara.sort()
        power_first_max.append(sort_ara[-2])
        power_second_max.append(sort_ara[-3])
        power_third_max.append(sort_ara[-4])
        index_first_max.append(ara.index(sort_ara[-2]))
        index_second_max.append(ara.index(sort_ara[-3]))
    feat_meal = pd.DataFrame()
    feat_meal['power_second_max'] = power_second_max
    feat_meal['power_third_max'] = power_third_max
    t = cleandata_inp.iloc[:, 22:25].idxmin(axis=1)
    maxi = cleandata_inp.iloc[:, 5:19].idxmax(axis=1)
    temp_list = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(cleandata_inp)):
        if i in maxi and i in t:
            temp_list.append(np.diff(cleandata_inp.iloc[:, maxi[i]:t[i]].iloc[i].tolist()).max())
            second_differential_data.append(
                np.diff(np.diff(cleandata_inp.iloc[:, maxi[i]:t[i]].iloc[i].tolist())).max())
        else:
            temp_list.append(0)
            second_differential_data.append(0)
        standard_deviation.append(np.std(cleandata_inp.iloc[i]))
    feat_meal['2ndDifferential'] = second_differential_data
    feat_meal['standard_deviation'] = standard_deviation
    return feat_meal


ft_mat = matrix_mlcreat(mealdta_df)
ft_mat1 = matrix_mlcreat(exmealdta)
ft_mat = pd.concat([ft_mat, ft_mat1]).reset_index().drop(columns='index')


def nomeal_matcreat(non_meal_data):
    inx_nonmel = non_meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 5).dropna().index
    cle_nomel = non_meal_data.drop(non_meal_data.index[inx_nonmel]).reset_index().drop(columns='index')
    cle_nomel = cle_nomel.interpolate(method='linear', axis=1)
    inx_dpga = cle_nomel.isna().sum(axis=1).replace(0, np.nan).dropna().index
    cle_nomel = cle_nomel.drop(cle_nomel.index[inx_dpga]).reset_index().drop(columns='index')
    feat_matnoml = pd.DataFrame()
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    for i, row in cle_nomel.iloc[:, 0:24].iterrows():
        ara1 = abs(rfft(row.values.tolist())).tolist()
        ara_st = abs(rfft(row.values.tolist())).tolist()
        ara_st.sort()
        power_first_max.append(ara_st[-2])
        power_second_max.append(ara_st[-3])
        power_third_max.append(ara_st[-4])
        index_first_max.append(ara1.index(ara_st[-2]))
        index_second_max.append(ara1.index(ara_st[-3]))
    feat_matnoml['power_second_max'] = power_second_max
    feat_matnoml['power_third_max'] = power_third_max
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    for i in range(len(cle_nomel)):
        if len(cle_nomel.iloc[:, 0:24].iloc[i].tolist()) > 1:
            first_differential_data.append(np.diff(cle_nomel.iloc[:, 0:24].iloc[i].tolist()).max())
        else:
            first_differential_data.append(0)

        if len(cle_nomel.iloc[:, 0:24].iloc[i].tolist()) > 2:
            second_differential_data.append(np.diff(np.diff(cle_nomel.iloc[:, 0:24].iloc[i].tolist())).max())
        else:
            second_differential_data.append(0)

        standard_deviation.append(np.std(cle_nomel.iloc[i]))
    feat_matnoml['2ndDifferential'] = second_differential_data
    feat_matnoml['standard_deviation'] = standard_deviation
    return feat_matnoml


feat_matnoml = nomeal_matcreat(dt_nomeal)
feat_matnoml1 = nomeal_matcreat(exdt_nomeal)
feat_matnoml = pd.concat([feat_matnoml, feat_matnoml1]).reset_index().drop(columns='index')

# Load the data and create labels
ft_mat['label'] = 1
feat_matnoml['label'] = 0
tot_dat = pd.concat([ft_mat, feat_matnoml]).reset_index().drop(columns='index')

# Shuffle the data
dat_et = shuffle(tot_dat, random_state=1).reset_index().drop(columns='index')
al_st = dat_et.drop(columns='label')
model_class = DecisionTreeClassifier(criterion="entropy")
X1, y1 = al_st, dat_et['label']
model_class.fit(X1, y1)
dump(model_class, 'RandomForestClassifier.pickle')

# Set up K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=108)

# Get the data and labels
Xa = dat_et.drop(columns='label')
ya = dat_et['label']

# Initialize an empty list to store scores
scores_rf = []
f1scro = []
psn = []
rlct = []


# Loop over K folds
def evaluate_model(X, y, n_folds):
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores_rf = []
    rlct = []
    f1scro = []
    psnu = []
    acc = []
    for train_index, test_index in kfold.split(X):
        # Get the training and testing data for this fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize a RandomForestClassifier and fit the training data
        classifier = RandomForestClassifier()
        classifier.fit(X_train, y_train)

        # Calculate the accuracy on the test data and store it
        score = classifier.score(X_test, y_test)
        y_pred = classifier.predict(X_test)

    scores_rf.append(score)
    rlct.append(metrics.recall_score(y_test, y_pred))
    f1scro.append(metrics.f1_score(y_test, y_pred))
    psnu.append(metrics.precision_score(y_test, y_pred))
    acc.append(metrics.accuracy_score(y_test, y_pred))

    return {'Accuracy': (scores_rf[0]) * 100, 'recall': (rlct[0]) * 100, 'f1': (f1scro[0]) * 100,
            'precision': (psnu[0]) * 100, 'overall_accuracy': (acc[0]) * 100}


results_list = evaluate_model(Xa, ya, n_folds=10)
print(results_list)
