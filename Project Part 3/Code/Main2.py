#!/usr/bin/env python
# coding: utf-8

# In[168]:


import pandas as pd
import numpy as np
from scipy.stats import entropy, iqr
from scipy.signal import periodogram
from sklearn.decomposition import PCA
from math import e
import warnings
from sklearn.preprocessing import normalize
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cluster
from scipy.fftpack import rfft
from scipy.integrate import simps
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns


# In[169]:


insln_dt = pd.read_csv("InsulinData.csv", usecols=["Date", "Time", "BWZ Carb Input (grams)"], index_col=False,)

cmgdf_dt = pd.read_csv("CGMData.csv", usecols=["Date", "Time", "Sensor Glucose (mg/dL)"])


# In[170]:


temp1 = []
for i, rows in cmgdf_dt.iterrows():
    dttime = rows["Date"] + " " + rows["Time"]
    dttiestmp = pd.to_datetime(dttime)
    temp1.append(dttiestmp)
cmgdf_dt["date_time_stamp"] = temp1

temp2 = []
for i, rows in insln_dt.iterrows():
    dttime = rows["Date"] + " " + rows["Time"]
    dttiestmp = pd.to_datetime(dttime)
    te2.append(dttiestmp)
insln_dt["date_time_stamp"] = temp2


# In[174]:


def extractMealData(cgm_df, insulin_df):
    cgmdfcp = cgm_df.copy()
    cgmdfcp = cgmdfcp.set_index('date_time_stamp')
    cgmdfcp = cgmdfcp.sort_index().reset_index()

    inslndfcp = insulin_df.copy()
    inslndfcp = inslndfcp.set_index('date_time_stamp')
    inslndfcp = inslndfcp.sort_index().dropna()

    inslndfcp = inslndfcp.replace(0.0, np.nan).reset_index().dropna()
    inslndfcp = inslndfcp.reset_index().drop(columns='index')

    lgitmltimelist = []
    lgitinsln = []
    i = 0
    while i < inslndfcp.shape[0] - 1:
        if (inslndfcp.iloc[i + 1]['date_time_stamp'] - inslndfcp.iloc[i]['date_time_stamp']).seconds >= 7200:
            lgitmltimelist.append(inslndfcp.iloc[i]['date_time_stamp'])
            lgitinsln.append(inslndfcp.iloc[i]['BWZ Carb Input (grams)'])
        i += 1

    lgitmltimelist.append(inslndfcp.iloc[inslndfcp.shape[0] - 1]['date_time_stamp'])
    lgitinsln.append(inslndfcp.iloc[inslndfcp.shape[0] - 1]['BWZ Carb Input (grams)'])
    mldta = []

    i = 0
    while i < len(lgitmltimelist):
        legit_meal_timestamp = lgitmltimelist[i]
        cgmmlval = cgmdfcp[(cgmdfcp['date_time_stamp'] >= legit_meal_timestamp - pd.Timedelta(minutes=30)) &
        (cgmdfcp['date_time_stamp'] <= legit_meal_timestamp + pd.Timedelta(minutes=120))]['Sensor Glucose (mg/dL)'].values.tolist()
        if len(cgmmlval) <= 30:
            cgmmlval = [np.nan] * (30 - len(cgmmlval)) + cgmmlval
        else:
            cgmmlval = cgmmlval[-30:]
        mldta.append(cgmmlval)
        i += 1

    mldf = pd.DataFrame(mldta)
    mldf = mldf[mldf.isna().sum(axis=1) <= 6].reset_index()

    lgitnonnainslin = []
    i = 0
    while i < len(mldf):
        lgitnonnainslin.append(lgitinsln[int(mldf.iloc[i]['index'])])
        ground_truth_val = []
        min_val = min(lgitnonnainslin)
        i += 1
    bins_count = (ceil((max(lgitnonnainslin) - min_val)/20))
    print()
    print("Bins Count :",bins_count)
    i = 0
    while i < len(lgitnonnainslin):
        ground_truth_val.append(int((lgitnonnainslin[i] - min_val) / 20))
        i += 1

    mldf = mldf.reset_index().drop(columns='index')
    mldf = mldf.interpolate(method='linear', axis=1, limit_direction='both')
    
    return mldf, np.asarray(ground_truth_val), lgitmltimelist, ground_truth_val


mldf, ground_truth_matrix, valid_times, ground_truth = extractMealData(cmgdf_dt, insln_dt)
mldf = mldf.drop('level_0', axis=1)
print()
print("Ground Truth Matrix")
print()
print(ground_truth)


# In[175]:


def row_entropy(row):
    v, cot = np.unique(row, return_counts=True)
    pbs = cot / len(row)
    entropy = 0
    i = 0
    while i < len(pbs):
        if pbs[i] != 0:
            entropy -= pbs[i] * np.log2(pbs[i])
        i += 1
    return entropy


# In[176]:


def createMealFeatureMat(inputMealdata):
    index = inputMealdata.isna().sum(axis=1).replace(0, np.nan).dropna().where(lambda x: x > 6).dropna().index
    mealdatacln = inputMealdata.drop(inputMealdata.index[index]).reset_index().drop(columns="index")
    mealdatacln = mealdatacln.interpolate(method="linear", axis=1)
    dropagn = mealdatacln.isna().sum(axis=1).replace(0, np.nan).dropna().index
    mealdatacln = mealdatacln.drop(inputMealdata.index[dropagn]).reset_index().drop(columns="index")
    mealdatacln = mealdatacln.dropna().reset_index().drop(columns="index")
    pwrfrstmtx, pwrscndmtx, pwrtirdmtx, idxfrstmtx, idxsndmtx, rmsvl, aucvl = ([], [], [], [], [], [], [])
    i = 0
    while i < len(mealdatacln):
        arr = abs(rfft(mealdatacln.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr = abs(rfft(mealdatacln.iloc[:, 0:30].iloc[i].values.tolist())).tolist()
        sndOrdArr.sort()
        pwrfrstmtx.append(sndOrdArr[-2])
        pwrscndmtx.append(sndOrdArr[-3])
        pwrtirdmtx.append(sndOrdArr[-4])
        idxfrstmtx.append(arr.index(sndOrdArr[-2]))
        idxsndmtx.append(arr.index(sndOrdArr[-3]))
        rmsrw = np.sqrt(np.mean(mealdatacln.iloc[i, 0:30] ** 2))
        rmsvl.append(rmsrw)
        auc_row = abs(simps(mealdatacln.iloc[i, 0:30], dx=1))
        aucvl.append(auc_row)
        i += 1
        
    ftrmelmat = pd.DataFrame()

    vloc = np.diff(mealdatacln, axis=1)
    vloc_min = np.min(vloc, axis=1)
    vloc_max = np.max(vloc, axis=1)
    vloc_mean = np.mean(vloc, axis=1)

    acc = np.diff(vloc, axis=1)
    acc_min = np.min(acc, axis=1)
    acc_max = np.max(acc, axis=1)
    acc_mean = np.mean(acc, axis=1)

    ftrmelmat['vloc_min'] = vloc_min
    ftrmelmat['vloc_max'] = vloc_max
    ftrmelmat['vloc_mean'] = vloc_mean
    ftrmelmat['acc_min'] = acc_min
    ftrmelmat['acc_max'] = acc_max
    ftrmelmat['acc_mean'] = acc_mean
    row_entropies = mealdatacln.apply(row_entropy, axis=1)
    ftrmelmat['row_entropies'] = row_entropies
    iqr_values = mealdatacln.apply(iqr, axis=1)
    ftrmelmat['iqr_values'] = iqr_values

    frtmx = []
    scnmtx = []
    thrdmtx = []
    fortmtx = []
    fivtmtx = []
    sxtmtx = []
    for it, rowdt in enumerate(mealdatacln.iloc[:, 0:30].values.tolist()):
        ara = abs(rfft(rowdt)).tolist()
        sort_ara = abs(rfft(rowdt)).tolist()
        sort_ara.sort()
        frtmx.append(sort_ara[-2])
        scnmtx.append(sort_ara[-3])
        thrdmtx.append(sort_ara[-4])
        fortmtx.append(sort_ara[-5])
        fivtmtx.append(sort_ara[-6])
        sxtmtx.append(sort_ara[-7])

    ftrmelmat['fft col1'] = frtmx
    ftrmelmat['fft col2'] = scnmtx
    ftrmelmat['fft col3'] = thrdmtx
    ftrmelmat['fft col4'] = fortmtx
    ftrmelmat['fft col5'] = fivtmtx
    ftrmelmat['fft col6'] = sxtmtx
    frequencies, psd_values = periodogram(mealdatacln, axis=1)

    psd1_values = np.mean(psd_values[:, 0:6], axis=1)
    psd2_values = np.mean(psd_values[:, 5:11], axis=1)
    psd3_values = np.mean(psd_values[:, 10:16], axis=1)
    ftrmelmat['psd1_values'] = psd1_values
    ftrmelmat['psd2_values'] = psd2_values
    ftrmelmat['psd3_values'] = psd3_values
    return ftrmelmat


# In[178]:


totaldata = createMealFeatureMat(mldf)
totaldata


# In[197]:


def sse_calculations(bin):
    if len(bin) != 0:
        avg = sum(bin) / len(bin)
        sse = sum([(i - avg) ** 2 for i in bin])
    else:
        sse = 0
    return sse

def calculate_entropy(y_true, y_pred, base=2):
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)
    Entropy = []
    i = 0
    while i < len(contingency_matrix):
        p = contingency_matrix[i, :]
        p = pd.Series(p).value_counts(normalize=True, sort=False)
        Entropy.append((-p / p.sum() * np.log(p / p.sum()) / np.log(2)).sum()) 
        i += 1

    TotalP = sum(contingency_matrix, 1);
    WholeEntropy = 0;
    i = 0
    while i < len(contingency_matrix):
        p = contingency_matrix[i, :]
        WholeEntropy = WholeEntropy + ((sum(p)) / (sum(TotalP))) * Entropy[i]
        i += 1
        
    return WholeEntropy

def calculate_purity_score(y_true, y_pred):
    contingency_matrix = cluster.contingency_matrix(y_true, y_pred)

    Purity = []
    i = 0
    while i < len(contingency_matrix):
        p = contingency_matrix[i, :]
        Purity.append(p.max() / p.sum())
        i += 1

    TotalP = sum(contingency_matrix, 1);
    WholePurity = 0;
    i = 0
    while i < len(contingency_matrix):
        p = contingency_matrix[i, :]
        WholePurity = WholePurity + ((sum(p)) / (sum(TotalP))) * Purity[i]
        i += 1

    return WholePurity


# In[204]:


def train_Kmeans_model(X_principal):
    clusterNum = 6
    k_means = KMeans(init="k-means++", n_clusters=clusterNum, n_init=12, random_state=42)
    k_means.fit(X_principal)
    kmeans_labels = k_means.labels_
    sse = k_means.inertia_

    # Categorize all the rows into clusters formed by K-means
    KMeans_Clusters = []
    bin = 0
    KMeans_Clusters = []
    while bin < 6:
        new = []
        i = 0
        while i < len(kmeans_labels):
            if kmeans_labels[i] == bin:
                new.append(i)
            i += 1
        KMeans_Clusters.append(new)
        bin += 1

    # Match K-means labels with ground truth labels and update the K-means labels
    def most_frequent(List):
        return max(set(List), key=List.count)

    Updated_kmeans_labels = kmeans_labels.copy()
    c = 0
    while c < 6:
        kmeans_cluster = KMeans_Clusters[c]
        true_labels = []
        i = 0
        while i < len(kmeans_cluster):
            val = kmeans_cluster[i]
            true_labels.append(ground_truth_matrix[val])
            i += 1
        updated_label = most_frequent(true_labels)
        i = 0
        while i < len(kmeans_cluster):
            val = kmeans_cluster[i]
            Updated_kmeans_labels[val] = updated_label
            i += 1    
        c += 1

    y_pred = k_means.fit_predict(X_principal)
    kmean_entropy = calculate_entropy(ground_truth_matrix, y_pred)
    kmean_purity_score = calculate_purity_score(ground_truth_matrix, y_pred)
    print("Confusion Matrix:")
    print()
    print(confusion_matrix(ground_truth_matrix,y_pred))

    return sse, kmean_entropy, kmean_purity_score


# In[205]:


Feature_Matrix_std = StandardScaler().fit_transform(tt)
Feature_Matrix_norm = normalize(Feature_Matrix_std)
Feature_Matrix_norm = pd.DataFrame(Feature_Matrix_norm)
pca = PCA(n_components=2)
X_principal = pca.fit_transform(Feature_Matrix_norm)
X_principal = pd.DataFrame(X_principal)
X_principal.columns = ['PCA1', 'PCA2']

kMeansSSE, kMeansEntropy, kMeansPurity = train_Kmeans_model(X_principal)


# In[206]:


def train_DBSCAN_model(X_principal):
    db = DBSCAN(eps=0.4, min_samples=7)
    db.fit(X_principal)
    db_labels = db.labels_
    unique_labels = set(db_labels) - {-1}
    data = pd.DataFrame()
    sse = 0
    for label in unique_labels:
        cluster = X_principal[db_labels == label]
        centroid = np.mean(cluster)
        sse += np.sum((cluster - centroid) ** 2)
    DBSCAN_Clusters = []
    for bin in range(-1, 6):
        new = []
        for i in range(0, len(db_labels)):
            if db_labels[i] == bin:
                new.append(i)
        DBSCAN_Clusters.append(new)

    def most_frequent(List):
        if not List:
            return None
        return max(set(List), key=List.count)

    Updated_dbscan_labels = db_labels.copy()
    c = 0
    while c < 7:
        db_cluster = DBSCAN_Clusters[c]
        true_labels = []
        i = 0
        while i < len(db_cluster):
            val = db_cluster[i]
            true_labels.append(ground_truth_matrix[val])
            i += 1
        updated_label = most_frequent(true_labels)
        # Update the dbscan labels
        i = 0
        while i < len(db_cluster):
            val = db_cluster[i]
            Updated_dbscan_labels[val] = updated_label
            i += 1
        c += 1

    data['cluster'] = Updated_dbscan_labels
    sse = data.groupby('cluster').apply(lambda x: ((x - data['cluster'].mean()) ** 2).sum()).sum()

    dbscan_entropy = calculate_entropy(ground_truth_matrix, Updated_dbscan_labels)
    dbscan_purity_score = calculate_purity_score(ground_truth_matrix, Updated_dbscan_labels)

    return sse, dbscan_entropy, dbscan_purity_score


# In[207]:


dbscanSSE, dbscanEntropy, dbscanPurity = train_DBSCAN_model(Feature_Matrix_norm)


# In[208]:


data = {
    'kMeansSSE': kMeansSSE,
    'dbscanSSE': dbscanSSE,
    'kMeansEntropy': kMeansEntropy,
    'dbscanEntropy': dbscanEntropy,
    'kMeansPurity': kMeansPurity,
    'dbscanPurity': dbscanPurity
}
results = pd.DataFrame(data)
results.to_csv('Results.csv', header=False, index=False)


# In[ ]:




