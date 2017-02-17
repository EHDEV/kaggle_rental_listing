# kaggle_rental_listing
import pandas as pd
import numpy as np
import os, sys
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
import re
import pdb
from collections import defaultdict
from sklearn.decomposition import PCA, NMF

data_train = pd.read_csv("../data/train_sample.csv")
sub_test = pd.read_csv("../data/test_sample.csv")

data_train.head(2)

np.where(pd.isnull(data.price))

def tointfloat(x, dtype=float):
    try:
        return float(x)
    except ValueError:
        pat = re.compile(pattern=r'[0-9]+')
        mat = pat.match(x)
        mat.group(0) if mat else np.nan
        
import time
def impute(col):
    imputer = Imputer(missing_values=-1, strategy="median", axis=0)
    return imputer.fit_transform(col)

def munge_bedbath(col):
    col = col.apply(tointfloat)
    col = col.fillna(-1)
    col = impute(col.reshape(-1,1)).reshape(-1)
    return col

def str_to_list(s):
    if type(s) == str:
#         res = s.lower().replace('[', '').replace(']', '').split(',')
        res = re.sub('[^A-Za-z0-9\,]+', '', s).lower().split(',')
    else: 
        res = []
    return res

def expand_features(col):
    col = col.apply(str_to_list)
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
        del new
    return ndf

def reduce_features(df):
    karray = np.array(df)
#     Impute np.nan values to 0
    imp = Imputer(strategy='most_frequent')
    karray = imp.fit_transform(karray)
#     Dimensionality reduction - PCA
    drf = PCA(n_components=20)
    karray = drf.fit_transform(karray)
    return pd.DataFrame(karray, columns=["pc%s" % str(x) for x in range(1,21)])


k = expand_features(data_train.features)

kcomps_ = reduce_features(k)
data_train = data_train.merge(kcomps_, left_index=True, right_index=True)
