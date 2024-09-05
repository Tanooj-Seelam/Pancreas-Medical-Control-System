#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data_CGM = pd.read_csv(
    "CGMData.csv", low_memory=False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])


# In[3]:


data_INSULIN = pd.read_csv(
    "InsulinData.csv", low_memory=False)


# In[4]:


print('Type of cgm data',type(data_CGM))
print('siz of cgm data',data_CGM.size)
print('Type of insulin data',type(data_INSULIN))
print('siz of insulin data',data_INSULIN.size)


# In[5]:


#Combing date and time to date_time_stamp column
data_CGM['date_time_stamp'] = pd.to_datetime(data_CGM['Date'] + ' ' + data_CGM['Time'])


# In[6]:


#Getting the null data
null_data = data_CGM[data_CGM['Sensor Glucose (mg/dL)'].isna()]['Date'].unique()


# In[7]:


#Data cleaning is done
data_CGM = data_CGM.set_index('Date').drop(index=null_data).reset_index()


# In[8]:


testdata_CGM = data_CGM.copy()


# In[9]:


testdata_CGM = testdata_CGM.set_index(pd.DatetimeIndex(data_CGM['date_time_stamp']))


# In[10]:


data_INSULIN['date_time_stamp'] = pd.to_datetime(data_INSULIN['Date'] + ' ' + data_INSULIN['Time'])


# In[11]:


#Getting the auto mode data by date time stamp
auto_mode_ACTIVE_df = data_INSULIN.sort_values(by='date_time_stamp', ascending=True)
auto_mode_ACTIVE = auto_mode_ACTIVE_df[auto_mode_ACTIVE_df['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]['date_time_stamp']


# In[12]:


auto_mode_data_df = data_CGM.sort_values(by='date_time_stamp', ascending=True)
auto_mode_data = auto_mode_data_df[auto_mode_data_df['date_time_stamp'] >= auto_mode_ACTIVE]


# In[13]:


manual_mode_data_df = data_CGM.sort_values(by='date_time_stamp', ascending=True)
manual_mode_data = manual_mode_data_df[manual_mode_data_df['date_time_stamp'] < auto_mode_ACTIVE]


# In[14]:


copy_auto_mode_data = auto_mode_data.copy()


# In[15]:


#Setting the index of the auto mode data to date time stamp column inorder to access the data based on it
auto_mode_data_set_index= copy_auto_mode_data.set_index('date_time_stamp')


# In[16]:


dates_list_df = auto_mode_data_set_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(
    lambda xn: xn > 0.8 * 288)
dates_list = dates_list_df.dropna().index.tolist()


# In[17]:


auto_mode_data_set_index = auto_mode_data_set_index.loc[auto_mode_data_set_index['Date'].isin(dates_list)]


# In[18]:


wholeday_automode_percent_df = auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_automode_percent = wholeday_automode_percent_df[
                                                       auto_mode_data_set_index[
                                                           'Sensor Glucose (mg/dL)'] > 180].groupby('Date')[
                                                       'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[19]:


daytime_automode_percentage_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                       ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_automode_percentage = daytime_automode_percentage_df[auto_mode_data_set_index[
                                                           'Sensor Glucose (mg/dL)'] > 180].groupby('Date')[
                                                       'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[20]:


overnight_automode_percentage_df = auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                        ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
overnight_automode_percentage = overnight_automode_percentage_df[auto_mode_data_set_index[
                                                             'Sensor Glucose (mg/dL)'] > 180].groupby('Date')[
                                                         'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[21]:


critical_wholday_automode_df =  auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_wholday_automode = critical_wholday_automode_df[auto_mode_data_set_index['Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                 'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[22]:


critical_daytime_automode_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                 ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_daytime_automode = critical_daytime_automode_df[auto_mode_data_set_index['Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                 'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[23]:


critical_overnight_automode_df = auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                 ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_overnight_automode = critical_overnight_automode_df[auto_mode_data_set_index['Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                 'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[24]:


range_wholeday_automode_df = auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_wholeday_automode = range_wholeday_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                            auto_mode_data_set_index[
                                                                'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                                'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[25]:


range_daytime_automode_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                               ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_daytime_automode = range_daytime_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                           auto_mode_data_set_index[
                                                               'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                               'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[26]:


range_overnight_automode_df = auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                 ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_overnight_automode = range_overnight_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                             auto_mode_data_set_index[
                                                                 'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                                 'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[27]:


range_sec_wholeday_automode_df = auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                    ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_wholeday_automode = range_sec_wholeday_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                                auto_mode_data_set_index[
                                                                    'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                    'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[28]:


range_sec_daytime_automode_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                   ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_daytime_automode = range_sec_daytime_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                               auto_mode_data_set_index[
                                                                   'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                   'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[29]:


range_sec_overnight_automode_df =  auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                     ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_overnight_automode = range_sec_overnight_automode_df[(auto_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                                 auto_mode_data_set_index[
                                                                     'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                     'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[30]:


wholeday_lv1_automode_df = auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                           ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_lv1_automode = wholeday_lv1_automode_df[(auto_mode_data_set_index[
                                                               'Sensor Glucose (mg/dL)'] < 70)].groupby('Date')[
                                                           'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[31]:


daytime_lv1_automode_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                          ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_lv1_automode = daytime_lv1_automode_df[auto_mode_data_set_index[
                                                              'Sensor Glucose (mg/dL)'] < 70].groupby('Date')[
                                                          'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[32]:


overnight_lv1_automode_df = auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                            ['Date', 'Time', 'Sensor Glucose (mg/dL)']] 
overnight_lv1_automode = overnight_lv1_automode_df[auto_mode_data_set_index[
                                                                'Sensor Glucose (mg/dL)'] < 70].groupby('Date')[
                                                            'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[33]:


wholeday_lv2_automode_df = auto_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                           ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_lv2_automode = wholeday_lv2_automode_df[auto_mode_data_set_index[
                                                               'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                           'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[34]:


daytime_lv2_automode_df = auto_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                          ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_lv2_automode = daytime_lv2_automode_df[auto_mode_data_set_index[
                                                              'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                          'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[35]:


overnight_lv2_automode_df = auto_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                            ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
overnight_lv2_automode = overnight_lv2_automode_df[auto_mode_data_set_index[
                                                                'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                            'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[36]:


copy_manual_mode_data = manual_mode_data.copy()
#Setting the index of the manual mode data to date time stamp column inorder to access the data based on it
manual_mode_data_set_index = copy_manual_mode_data.set_index('date_time_stamp')


# In[37]:


date_list2_df = manual_mode_data_set_index.groupby('Date')['Sensor Glucose (mg/dL)'].count().where(
    lambda x: x > 0.8 * 288)
date_list2 = date_list2_df.dropna().index.tolist()


# In[38]:


manual_mode_data_set_index = manual_mode_data_set_index.loc[manual_mode_data_set_index['Date'].isin(date_list2)]


# In[39]:


wholeday_manual_df = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                      ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_manual = wholeday_manual_df[manual_mode_data_set_index['Sensor Glucose (mg/dL)'] > 180].groupby(
     'Date')['Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[40]:


daytime_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                     ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_manual = daytime_manual_df[ manual_mode_data_set_index['Sensor Glucose (mg/dL)'] > 180].groupby(
     'Date')['Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[41]:


overnight_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                       ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
overnight_manual = overnight_manual_df[manual_mode_data_set_index[
                                                           'Sensor Glucose (mg/dL)'] > 180].groupby('Date')[
                                                       'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[42]:


critical_wholeday_manual_df = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                               ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_wholeday_manual = critical_wholeday_manual_df[manual_mode_data_set_index[
                                                                   'Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                                                               'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[43]:


critical_daytime_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                              ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_daytime_manual = critical_daytime_manual_df[manual_mode_data_set_index[
                                                                  'Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                                                              'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[44]:


critical_overnight_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                 ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
critical_overnight_manual = critical_overnight_manual_df[ manual_mode_data_set_index['Sensor Glucose (mg/dL)'] > 250].groupby('Date')[
                 'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[45]:


range_wholeday_manual_df  = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                              ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_wholeday_manual = range_wholeday_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                          manual_mode_data_set_index[
                                                              'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                              'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[46]:


range_daytime_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                             ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_daytime_manual = range_daytime_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                         manual_mode_data_set_index[
                                                             'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                             'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[47]:


range_overnight_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                               ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_overnight_manual = range_overnight_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                           manual_mode_data_set_index[
                                                               'Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')[
                                               'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[48]:


range_sec_wholeday_manual_df = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                  ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_wholeday_manual = range_sec_wholeday_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                              manual_mode_data_set_index[
                                                                  'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                  'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[49]:


range_sec_daytime_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                 ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_daytime_manual  = range_sec_daytime_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                             manual_mode_data_set_index[
                                                                 'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                 'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[50]:


range_sec_overnight_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                   ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
range_sec_overnight_manual = range_sec_overnight_manual_df[(manual_mode_data_set_index['Sensor Glucose (mg/dL)'] >= 70) & (
                                                               manual_mode_data_set_index[
                                                                   'Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')[
                                                   'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[51]:


wholeday_lv1_manual_df = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                         ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_lv1_manual = wholeday_lv1_manual_df[manual_mode_data_set_index[
                                                             'Sensor Glucose (mg/dL)'] < 70].groupby('Date')[
                                                         'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[52]:


daytime_lv1_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                        ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_lv1_manual = daytime_lv1_manual_df[manual_mode_data_set_index[
                                                            'Sensor Glucose (mg/dL)'] < 70].groupby('Date')[
                                                        'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[53]:


overnight_lv1_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                          ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
overnight_lv1_manual = overnight_lv1_manual_df[manual_mode_data_set_index[
                                                              'Sensor Glucose (mg/dL)'] < 70].groupby('Date')[
                                                          'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[54]:


wholeday_lv2_manual_df = manual_mode_data_set_index.between_time('0:00:00', '23:59:59')[
                                                         ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
wholeday_lv2_manual = wholeday_lv2_manual_df[manual_mode_data_set_index[
                                                             'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                         'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[55]:


daytime_lv2_manual_df = manual_mode_data_set_index.between_time('6:00:00', '23:59:59')[
                                                        ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
daytime_lv2_manual = daytime_lv2_manual_df[manual_mode_data_set_index[
                                                            'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                        'Sensor Glucose (mg/dL)'].count() / 288 * 100    


# In[56]:


overnight_lv2_manual_df = manual_mode_data_set_index.between_time('0:00:00', '05:59:59')[
                                                          ['Date', 'Time', 'Sensor Glucose (mg/dL)']]
overnight_lv2_manual = overnight_lv2_manual_df[manual_mode_data_set_index[
                                                              'Sensor Glucose (mg/dL)'] < 54].groupby('Date')[
                                                          'Sensor Glucose (mg/dL)'].count() / 288 * 100


# In[57]:


results_df = pd.DataFrame({'percent_time_in_hyperglycemia_overnight': [
    overnight_manual.mean(axis=0),
    overnight_automode_percentage.mean(axis=0)],

                           'percent_time_in_hyperglycemia_critical_overnight': [
                               critical_overnight_manual.mean(axis=0),
                               critical_overnight_automode.mean(axis=0)],

                           'percent_time_in_range_overnight': [range_overnight_manual.mean(axis=0),
                                                               range_overnight_automode.mean(axis=0)],

                           'percent_time_in_range_sec_overnight': [
                               range_sec_overnight_manual.mean(axis=0),
                               range_sec_overnight_automode.mean(axis=0)],

                           'percent_time_in_hypoglycemia_lv1_overnight': [
                               overnight_lv1_manual.mean(axis=0),
                              overnight_lv1_automode.mean(axis=0)],

                           'percent_time_in_hypoglycemia_lv2_overnight': [
                               np.nan_to_num(overnight_lv2_manual.mean(axis=0)),
                               overnight_lv2_automode.mean(axis=0)],
                           'percent_time_in_hyperglycemia_daytime': [
                               daytime_manual.mean(axis=0),
                               daytime_automode_percentage.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_daytime': [
                               critical_daytime_manual.mean(axis=0),
                               critical_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_daytime': [range_daytime_manual.mean(axis=0),
                                                             range_daytime_automode.mean(axis=0)],
                           'percent_time_in_range_sec_daytime': [range_sec_daytime_manual.mean(axis=0),
                                                                range_sec_daytime_automode.mean(
                                                                     axis=0)],
                           'percent_time_in_hypoglycemia_lv1_daytime': [
                               daytime_lv1_manual.mean(axis=0),
                               daytime_lv1_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_daytime': [
                               daytime_lv2_manual.mean(axis=0),
                               daytime_lv2_automode.mean(axis=0)],

                           'percent_time_in_hyperglycemia_wholeday': [
                               wholeday_manual.mean(axis=0),
                               wholeday_automode_percent.mean(axis=0)],
                           'percent_time_in_hyperglycemia_critical_wholeday': [
                               critical_wholeday_manual.mean(axis=0),
                               critical_wholday_automode.mean(axis=0)],
                           'percent_time_in_range_wholeday': [range_wholeday_manual.mean(axis=0),
                                                              range_wholeday_automode.mean(axis=0)],
                           'percent_time_in_range_sec_wholeday': [
                               range_sec_wholeday_manual.mean(axis=0),
                               range_sec_wholeday_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv1_wholeday': [
                               wholeday_lv1_manual.mean(axis=0),
                               wholeday_lv1_automode.mean(axis=0)],
                           'percent_time_in_hypoglycemia_lv2_wholeday': [
                               wholeday_lv2_manual.mean(axis=0),
                               wholeday_lv2_automode.mean(axis=0)]

                           },
                          index=['manual_mode', 'auto_mode'])


# In[58]:


results_df.to_csv('Results.csv',
                  header=False, index=False)


# In[ ]:




