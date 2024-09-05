import pandas as pd
import datetime as dt
import time
import math
# ###importing data files
full_ins_data = pd.read_csv('InsulinData.csv')
data_ins = full_ins_data[['Date', 'Time', 'Alarm']]
full_cgm_data = pd.read_csv('CGMData.csv')
data_cgm = full_cgm_data[['Date', 'Time', 'Sensor Glucose (mg/dL)']]

# ### creating date time object

data_ins['DateTime'] = pd.to_datetime(data_ins['Date'] + " " + data_ins['Time'], format = '%m/%d/%Y %H:%M:%S')

data_cgm['DateTime'] = pd.to_datetime(data_cgm['Date'] + " " + data_cgm['Time'], format = '%m/%d/%Y %H:%M:%S')


# ### Checking out when the auto mode is turned on based on the alarm code

auto_datetime = data_ins[data_ins['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF']['DateTime'].min()


# ### Divding the CGM data based on auto and manual modes

cgmADf = data_cgm[data_cgm['DateTime'] >= auto_datetime]

cgmMDf = data_cgm[data_cgm['DateTime'] < auto_datetime]

# ### Dividing the data into different timeframes of the day

cgmADayDf = cgmADf[cgmADf['DateTime'].dt.hour >= 6]

cgmANightDf = cgmADf[cgmADf['DateTime'].dt.hour < 6]

cgmMDayDf = cgmMDf[cgmMDf['DateTime'].dt.hour >= 6]

cgmMNightDf = cgmMDf[cgmMDf['DateTime'].dt.hour < 6]


# #### Removing the dates where the entries are not more than 70% of the expected count

def drop_dates(df, couCol, threshold = 0, exp_count = 288):
    groupDataCount = df.groupby('Date').count()[couCol]
    keys_drop = list(groupDataCount[(groupDataCount / exp_count) < threshold].keys())
    result = df[~df['Date'].isin(keys_drop)]
    return result

threshold = 0


autoDf = drop_dates(df = cgmADf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 288)

manualDf = drop_dates(df = cgmMDf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 288)


autoDayDf = drop_dates(df = cgmADayDf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 216)

manualDayDf = drop_dates(df = cgmMDayDf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 216)


autoNightDf = drop_dates(df = cgmANightDf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 72)

manualNightDf = drop_dates(df = cgmMNightDf, couCol = 'Sensor Glucose (mg/dL)', threshold = threshold, exp_count = 72)



def get_percentage_of_entries(dframe, colName, interval, exp_count_per_day):
    df = dframe
    numDays = len(df['Date'].unique())
    outOf = numDays * exp_count_per_day
    ran_entries = 0
    (minRan, maxRan) = interval
    if minRan is not None and maxRan is not None:
        ran_entries = df[(df[colName] >= minRan) & (df[colName] <= maxRan)].count()[colName]
    elif minRan is not None:
        ran_entries = df[df[colName] > minRan].count()[colName]
    elif maxRan is not None:
        ran_entries = df[df[colName] < maxRan].count()[colName]
    return (ran_entries / (outOf * 1.0)) * 100

colName = 'Sensor Glucose (mg/dL)'

autoDfList = [(autoNightDf, 288), (autoDayDf, 288), (autoDf, 288)]
manualDfList = [(manualNightDf, 288), (manualDayDf, 288), (manualDf, 288)]


interList = [(180, None), (250, None), (70, 180), (70, 150), (None, 70), (None, 54)]

manualEntries = []
for df,exp_count_per_day in manualDfList:
    for inter in interList:
        manualEntries.append(get_percentage_of_entries(df, colName, inter, exp_count_per_day))
autoEntries = []
for df,exp_count_per_day in autoDfList:
    for inter in interList:
        autoEntries.append(get_percentage_of_entries(df, colName, inter, exp_count_per_day))
manualEntries.append(1.1)
autoEntries.append(1.1)

# ### The results are then formatted to csv file
res_df = pd.DataFrame([manualEntries, autoEntries])

res_df.to_csv('Result.csv', index = False, header = False)
