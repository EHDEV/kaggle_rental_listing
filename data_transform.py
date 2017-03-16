# Data Preprocessing
%matplotlib inline
import seaborn as sns
import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import re
import pdb
from collections import defaultdict
from sklearn.decomposition import PCA, NMF
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, LabelEncoder, LabelBinarizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import shuffle
# data_train.head(2)

# np.where(pd.isnull(data.price))

def split_test_train(X, y, test_size=0.2, seed=100):
    """
    Randomly split the data into test and train sets
    :param data:
    :param test_percentage:
    :return: list of indices for the test set
    """
    np.random.seed(seed)
    X, y = shuffle(X, y, random_state=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, y_train, X_test, y_test


def tointfloat(x, dtype=float):
    try:
        return dtype(x)
    except:
        pat = re.compile(pattern=r'[0-9]+')
        mat = pat.match(x)
        if mat:
            return mat.group(0) 
        else: 
            return np.nan

def to_float_hard(x):
    try:
        return float(x)
    except:
        return 0

def scale_variable(var, sd=False):
    if sd:
        var = (var - var.mean())/var.std()
    else:
        var = (var - var.min())/(var.max()-var.min())
    
    return var

#     pdb.set_trace()
#     if std:
#         var = StandardScaler().fit_transform(var.reshape(1, -1))
#     var = MinMaxScaler().fit_transform(var.reshape(1,-1))
#     return var.reshape(-1)

def conv_geo_tocx(df):
    # StandardScaler prefers a Matrix instead of a vector so
    lat = df['latitude'].apply(to_float_hard).fillna(0).values
    lon = df['longitude'].apply(to_float_hard).fillna(0).values
    
    R = 1
    df['xcx'] = MinMaxScaler().fit_transform(R * np.cos(lat) * np.cos(lon))
    df['ycx'] = MinMaxScaler().fit_transform(R * np.cos(lat) * np.sin(lon))
    df['zcx'] = MinMaxScaler().fit_transform(R * np.sin(lat))
    
    
    return df
        
import time
def impute(col):
    imputer = Imputer(missing_values=-1, strategy="median", axis=0)
    return imputer.fit_transform(col)

def munge_bedbath(col):
    col = col.fillna(-1)
    col = col.apply(tointfloat)
    col = impute(col.reshape(-1,1)).reshape(-1)
    return col.astype(float)



def str_to_list(s):
    if type(s) in [str, unicode]:
#         res = s.lower().replace('[', '').replace(']', '').split(',')
        res = re.sub('([^A-Za-z0-9\, *|;]+|http.+jpg)', '', s).lower()
        res = re.sub("( the | is | this | there | with | a | are | for | from | which | of | to | not | in |)", '', res)
        res = re.sub("(''+,\s*|''|^[\s',]{0,})", "", res)
        res = re.sub('([\*,;|]|\s{2,})', ',', res)
        res = re.sub("""['"a-zA-Z0-9]+[0-9]{3,}[a-zA-Z0-9'"]*""", '', res)
        res = re.sub("('\s'*|''|^[\s',]{0,}),*", ",", res)
        res = re.sub("u[^a-zA-Z\s]+(?![a-zA-Z])", ",", res)
        res = res.split(',')
        try:
            slen = min(max(len(res) * 1/2, 6), len(res))
            idx = np.random.choice(len(res), slen, replace=False)
            res = list(np.array(res)[idx])
        except:
            pdb.set_trace()
    elif type(s) != list: 
        res = []
    else:
        res = s
    return res

def split_to_len(x):
    try:
        
        l = len(x.split(','))
    except:
        l = 0
        
    return l
        

def expand_features(col):
    import time
    start_time = time.time()
    ndf = pd.DataFrame()
    new_cols = set()
    for i, row in enumerate(col):
        if row > [] and len(row) > 1:
            new_cols = new_cols.union(row)
    print time.time()
    for idx, row in enumerate(col):
        new = defaultdict(list)
        exc = new_cols.difference(row) 
        if row > [] and len(row) > 1:
            for nc in set(row):
                new[nc].append(1)
        for ec in exc:
            new[ec].append(0)
        try:
            tmp = pd.DataFrame(new)
            ndf = pd.concat((ndf, tmp), axis=0)
        except Exception, e:
            print e
            print '**********'
            continue
        del new
    print("--- %s seconds ---" % (time.time() - start_time))
    return ndf

def expand_features2(col):
    import time
    start_time = time.time()
    new_cols = set()
    for i, row in enumerate(col):
        if row > [] and len(row[0]) > 1:
            new_cols = new_cols.union(row)
    ndf = pd.DataFrame(columns=new_cols)
    print (len(new_cols))
    for i,row in enumerate(col):
    #     ooo = LabelBinarizer()
    #     val = ooo.fit_transform(row).T.sum(axis=0)
        try:
            if len(row) > 0 and row[0] > '': 
                ndf.loc[i, row] = 1 
            else:
                ndf.loc[i] = 0
        except Exception as e:
            print e
            continue
    print("--- %s seconds ---" % (time.time() - start_time))
    return ndf


def reduce_features(df, ncomps=40):
    karray = np.array(df)
#     Impute np.nan values to 0
    imp = Imputer(strategy='most_frequent')
    karray = imp.fit_transform(karray)
#     Dimensionality reduction - PCA
    drf = PCA(n_components=ncomps)
    karray = drf.fit_transform(karray)
    return pd.DataFrame(karray, columns=["pc%s" % str(x) for x in range(1,ncomps+1)])

def dummiefy(var):
    dummified = pd.get_dummies(var)
    dummified.columns = [
        var.name + "_" + str(int(c)) if type(c) != str else var.name + "_" + str(c) for c in dummified.columns
        ]
    return dummified


def encode_target(target):
    le = LabelEncoder()
    le.fit(target)
    return le.transform(target), list(le.classes_)

def is_outlier(col):
    iq_range = col.quantile(0.75) - col.quantile(0.25)
    median = col.median()
    try:
        return col > abs(median + (1.5 * iq_range))
    except:
        return col.fillna(0) > abs(median + (1.5 * iq_range))
