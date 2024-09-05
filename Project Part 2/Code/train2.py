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

dfinsuline = pd.read_csv('InsulinData.csv', low_memory=False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

dfcmgdf = pd.read_csv('CGMData.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

dfinsuline['date_time_stamp'] = pd.to_datetime(dfinsuline['Date'] + ' ' + dfinsuline['Time'])
dfcmgdf['date_time_stamp'] = pd.to_datetime(dfcmgdf['Date'] + ' ' + dfcmgdf['Time'])

dfinsuline_alter = pd.read_csv('Insulin_patient2.csv', low_memory=False,usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

dfcmgdf_alter = pd.read_csv('CGM_patient2.csv', low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])

dfinsuline_alter['date_time_stamp'] = pd.to_datetime(dfinsuline_alter['Date'] + ' ' + dfinsuline_alter['Time'])
dfcmgdf_alter['date_time_stamp'] = pd.to_datetime(dfcmgdf_alter['Date'] + ' ' + dfcmgdf_alter['Time'])


def mealdata_creation(dfinsuline, dfcmgdf, identifydate):
    # Create a copy of the insulin dataframe and set its index to the date_time_stamp column
    temp_dfinsuline = dfinsuline.copy()
    temp_dfinsuline = temp_dfinsuline.set_index('date_time_stamp')

    # Sort the dataframe by date_time_stamp in ascending order, drop rows with missing values, and reset the index
    valid_carb_timestamps_df = temp_dfinsuline.sort_values(by='date_time_stamp', ascending=True).dropna().reset_index()

    # Replace 0.0 values in the BWZ Carb Input (grams) column with NaN, drop rows with missing values, and reset the
    # index
    valid_carb_timestamps_df['BWZ Carb Input (grams)'].replace(0.0, np.nan, inplace=True)
    valid_carb_timestamps_df = valid_carb_timestamps_df.dropna().reset_index().drop(columns='index')

    # Create a list of valid timestamps that are at least 2.5 hours apart
    timestamp_validlist = []
    value = 0
    for idx, i in enumerate(valid_carb_timestamps_df['date_time_stamp']):
        try:
            value = (valid_carb_timestamps_df['date_time_stamp'][idx + 1] - i).seconds / 60.0
            if value >= 120:
                timestamp_validlist.append(i)
        except KeyError:
            break

    # Create a list to store the glucose values for each meal
    meal_glucose_list = []

    # Loop through the list of valid timestamps and extract the glucose values for each meal
    for idx, i in enumerate(timestamp_validlist):
        start = pd.to_datetime(i - timedelta(minutes=30))
        end = pd.to_datetime(i + timedelta(minutes=120))
        if identifydate == 1:
            get_date = i.date().strftime('%m/%d/%Y')
        else:
            get_date = i.date().strftime('%Y-%m-%d')
        glucose_values = dfcmgdf.loc[dfcmgdf['Date'] == get_date].set_index('date_time_stamp').between_time(
            start_time=start.strftime('%H:%M:%S'), end_time=end.strftime('%H:%M:%S'))[
            'Sensor Glucose (mg/dL)'].values.tolist()
        meal_glucose_list.append(glucose_values)

    # Convert the list of glucose values to a dataframe and return it
    return pd.DataFrame(meal_glucose_list)

meal_data = mealdata_creation(dfinsuline, dfcmgdf, 1)
meal_data1 = mealdata_creation(dfinsuline_alter, dfcmgdf_alter, 2)

# Create a list to store the meal data
meal_data_list = []
meal_data_list.append(meal_data.iloc[:, 0:24].values.tolist())
meal_data_list.append(meal_data1.iloc[:, 0:24].values.tolist())

# Concatenate the meal data from all patients into a single dataframe
final_meal_data = pd.DataFrame(np.concatenate(meal_data_list, axis=0))

# Rename the columns of the final dataframe
new_cols = {i: f"Meal_{i+1}" for i in range(final_meal_data.shape[1])}
final_meal_data = final_meal_data.rename(columns=new_cols)


# Create meal data for dfinsuline and dfcmgdf with group number 1
meal_data = mealdata_creation(dfinsuline, dfcmgdf, 1)

# Create meal data for dfinsuline_alter and dfcmgdf_alter with group number 2
meal_data1 = mealdata_creation(dfinsuline_alter, dfcmgdf_alter, 2)


### No-Meal data extraction

def createnomealdata(dfinsuline, dfcmgdf):
    insulin_no_meal_df = dfinsuline.copy()
    test1_df = insulin_no_meal_df.sort_values(by='date_time_stamp', ascending=True).replace(0.0, np.nan).dropna().copy()
    test1_df = test1_df.reset_index().drop(columns='index')
    valid_timestamp = []
    idx = 0
    while True:
        try:
            i = test1_df['date_time_stamp'][idx]
            value = (test1_df['date_time_stamp'][idx + 1] - i).seconds // 3600
            if value >= 4:
                valid_timestamp.append(i)
            idx += 1
        except KeyError:
            break

    dataset = []
    idx = 0
    while idx < len(valid_timestamp) - 1:
        i = valid_timestamp[idx]
        iteration_dataset = 1
        length_of_24_dataset = len(dfcmgdf.loc[(dfcmgdf['date_time_stamp'] >= i + pd.Timedelta(hours=2)) & (
                    dfcmgdf['date_time_stamp'] < valid_timestamp[idx + 1])]) // 24
        while iteration_dataset <= length_of_24_dataset:
            if iteration_dataset == 1:
                dataset.append(dfcmgdf.loc[(dfcmgdf['date_time_stamp'] >= i + pd.Timedelta(hours=2)) & (
                            dfcmgdf['date_time_stamp'] < valid_timestamp[idx + 1])]['Sensor Glucose (mg/dL)'][
                               :iteration_dataset * 24].values.tolist())
            else:
                dataset.append(dfcmgdf.loc[(dfcmgdf['date_time_stamp'] >= i + pd.Timedelta(hours=2)) & (
                            dfcmgdf['date_time_stamp'] < valid_timestamp[idx + 1])]['Sensor Glucose (mg/dL)'][
                               (iteration_dataset - 1) * 24:iteration_dataset * 24].values.tolist())
            iteration_dataset += 1
        idx += 1

    return pd.DataFrame(dataset)


no_meal_data_list = []
for df_insuline, df_cmgdf in [(dfinsuline, dfcmgdf), (dfinsuline_alter, dfcmgdf_alter)]:
    no_meal_data = createnomealdata(df_insuline, df_cmgdf)
    no_meal_data_list.append(no_meal_data)


def meal_matrix_creation(input_mealdata):
    idx = input_mealdata.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    input_data_cleaned = input_mealdata.drop(input_mealdata.index[idx]).reset_index().drop(columns='index')
    input_data_cleaned = input_data_cleaned.interpolate(method='linear', axis=1)
    index_to_drop_again = input_data_cleaned.isna().sum(axis=1).replace(0, np.nan).dropna().index
    input_data_cleaned = input_data_cleaned.drop(input_mealdata.index[index_to_drop_again]).reset_index().drop(
        columns='index')
    input_data_cleaned = input_data_cleaned.dropna().reset_index().drop(columns='index')
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    i = 0
    while i < len(input_data_cleaned):
        arr = abs(rfft(input_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sod_arr = abs(rfft(input_data_cleaned.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sod_arr.sort()
        power_first_max.append(sod_arr[-2])
        power_second_max.append(sod_arr[-3])
        power_third_max.append(sod_arr[-4])
        index_first_max.append(arr.index(sod_arr[-2]))
        index_second_max.append(arr.index(sod_arr[-3]))
        i += 1
    featured_mealmat = pd.DataFrame()
    featured_mealmat['power_second_max'] = power_second_max
    featured_mealmat['power_third_max'] = power_third_max
    t = input_data_cleaned.iloc[:, 22:25].idxmin(axis=1)
    maxi = input_data_cleaned.iloc[:, 5:19].idxmax(axis=1)
    temp_list = []
    second_differential_data = []
    standard_deviation = []
    i = 0
    while i < len(input_data_cleaned):
        temp_list.append(np.diff(input_data_cleaned.iloc[:, maxi[i]:t[i]].iloc[i].tolist()).max())
        second_differential_data.append(
            np.diff(np.diff(input_data_cleaned.iloc[:, maxi[i]:t[i]].iloc[i].tolist())).max())
        standard_deviation.append(np.std(input_data_cleaned.iloc[i]))
        i += 1
    featured_mealmat['2ndDifferential'] = second_differential_data
    featured_mealmat['standard_deviation'] = standard_deviation
    return featured_mealmat


feat_matmeal = meal_matrix_creation(meal_data)
feat_matmeal1 = meal_matrix_creation(meal_data1)
feat_matmeal = pd.concat([feat_matmeal, feat_matmeal1]).reset_index().drop(columns='index')


def nomeal_matrix_creation(non_meal_data):
    inx_non_meal = non_meal_data.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 5).dropna().index
    cleaned_nomeal = non_meal_data.drop(non_meal_data.index[inx_non_meal]).reset_index().drop(columns='index')
    cleaned_nomeal = cleaned_nomeal.interpolate(method='linear', axis=1)
    idx_dpaga = cleaned_nomeal.isna().sum(axis=1).replace(0, np.nan).dropna().index
    cleaned_nomeal = cleaned_nomeal.drop(cleaned_nomeal.index[idx_dpaga]).reset_index().drop(columns='index')
    featureed_matnomeal = pd.DataFrame()
    power_first_max = []
    index_first_max = []
    power_second_max = []
    index_second_max = []
    power_third_max = []
    i = 0
    while i < len(cleaned_nomeal):
        arr1 = abs(rfft(cleaned_nomeal.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        arr_std = abs(rfft(cleaned_nomeal.iloc[:, 0:24].iloc[i].values.tolist())).tolist()
        arr_std.sort()
        power_first_max.append(arr_std[-2])
        power_second_max.append(arr_std[-3])
        power_third_max.append(arr_std[-4])
        index_first_max.append(arr1.index(arr_std[-2]))
        index_second_max.append(arr1.index(arr_std[-3]))
        i += 1
    featureed_matnomeal['power_second_max'] = power_second_max
    featureed_matnomeal['power_third_max'] = power_third_max
    first_differential_data = []
    second_differential_data = []
    standard_deviation = []
    i = 0
    while i < len(cleaned_nomeal):
        first_differential_data.append(np.diff(cleaned_nomeal.iloc[:, 0:24].iloc[i].tolist()).max())
        second_differential_data.append(np.diff(np.diff(cleaned_nomeal.iloc[:, 0:24].iloc[i].tolist())).max())
        standard_deviation.append(np.std(cleaned_nomeal.iloc[i]))
        i += 1
    featureed_matnomeal['2ndDifferential'] = second_differential_data
    featureed_matnomeal['standard_deviation'] = standard_deviation
    return featureed_matnomeal


featureed_matnomeal = nomeal_matrix_creation(no_meal_data_list[0])
featureed_matnomeal1 = nomeal_matrix_creation(no_meal_data_list[1])
featureed_matnomeal = pd.concat([featureed_matnomeal, featureed_matnomeal1]).reset_index().drop(columns='index')

# Load the data and create labels
feat_matmeal['label'] = 1
featureed_matnomeal['label'] = 0
total_data = pd.concat([feat_matmeal, featureed_matnomeal]).reset_index().drop(columns='index')

# Shuffle the data
dataset = shuffle(total_data, random_state=1).reset_index().drop(columns='index')
atl_dst = dataset.drop(columns='label')
model = DecisionTreeClassifier(criterion="entropy")
X1, y1 = atl_dst, dataset['label']
model.fit(X1, y1)
dump(model, 'RandomForestClassifier.pickle')

# Set up K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=108)

# Get the data and labels
X = dataset.drop(columns='label')
y = dataset['label']

# Initialize an empty list to store scores
scores_rf = []
f1scr = []
psn = []
rlc = []

# Loop over K folds
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
    rlc.append(metrics.recall_score(y_test, y_pred))
    f1scr.append(metrics.f1_score(y_test, y_pred))
    psn.append(metrics.precision_score(y_test, y_pred))
    rlc.append(metrics.accuracy_score(y_test, y_pred))

# Print the average accuracy over all folds
print(f'RandomForestClassifier accuracy: {(sum(scores_rf) / len(scores_rf) * 100)}')
print("Recall : ", np.mean(rlc) * 100)
print("F1 Score : ", np.mean(f1scr) * 100)
print("Precision : ", np.mean(psn) * 100)
