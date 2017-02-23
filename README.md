# kaggle_rental_listing

##### 1

# Data Preprocessing

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
from sklearn.model_selection import train_test_split



data_train = pd.read_csv("../data/train_sample.csv")
sub_test = pd.read_csv("../data/test_sample.csv")

# data_train.head(2)

# np.where(pd.isnull(data.price))

def split_test_train(X, y, test_size=0.2):
    """
    Randomly split the data into test and train sets
    :param data:
    :param test_percentage:
    :return: list of indices for the test set
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)
    return X_train, y_train, X_test, y_test


def tointfloat(x, dtype=float):
    try:
        return dtype(x)
    except ValueError:
        pat = re.compile(pattern=r'[0-9]+')
        mat = pat.match(x)
        mat.group(0) if mat else np.nan

def to_float_hard(x):
    try:
        return float(x)
    except ValueError:
        return 0
        
import time
def impute(col):
    imputer = Imputer(missing_values=-1, strategy="median", axis=0)
    return imputer.fit_transform(col)

def munge_bedbath(col):
    col = col.apply(tointfloat)
    col = col.fillna(-1)
    col = impute(col.reshape(-1,1)).reshape(-1)
    return col.astype(float)

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


def reduce_features(df, ncomps=20):
    karray = np.array(df)
#     Impute np.nan values to 0
    imp = Imputer(strategy='most_frequent')
    karray = imp.fit_transform(karray)
#     Dimensionality reduction - PCA
    drf = PCA(n_components=20)
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

k = expand_features(data_train.features)
kcomps_ = reduce_features(k)

data_clean = data_train.copy()

data_clean = data_train.merge(kcomps_, left_index=True, right_index=True)
data_clean.drop(['Unnamed: 0'], axis=1, inplace=True)
data_clean['number_of_photos'] = data_clean.photos.apply(str_to_list).apply(len)
data_clean['bedrooms'] = munge_bedbath(data_clean.bedrooms)
data_clean['bathrooms'] = munge_bedbath(data_clean.bathrooms)
data_clean['latitude'] = data_clean['latitude'].apply(to_float_hard)

    
il_fix = []
iremove = []
for i, d in enumerate(data_train.interest_level):
    if len(str(d)) > 6:
        if data_train.building_id.loc[i] in ['medium', 'high', 'low']:
            il_fix += [i]
        else: 
            iremove += [i]
    elif str(d).lower() in ['nan', '']:
        iremove += [i]
        
data_clean.drop(iremove, inplace=True)
data_clean.interest_level.loc[il_fix] = data_clean.building_id.loc[il_fix]
data_clean.drop(data_clean[~data_clean.interest_level.isin(['medium', 'high', 'low'])].index, inplace=True)
data_clean['interest_level'], _classes = encode_target(data_clean.interest_level)
data_clean = data_clean[~data_clean.price.isnull()][~data_clean.longitude.isnull()]

exclude_columns = [
    'building_id', 'created', 'description', 'display_address', 'features', 'listing_id', 'manager_id', 'photos', 'street_address'
]

for col in exclude_columns:
    del data_clean[col]


######## 2


# Feature selection, Model Selection, training, hyperparameter tuning, prediction
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
def train_model(X_train, y_train, clfs=[]):
    """
    Cross validate and train on best model
    :param clf:
    :return:
    """
    clf_scores = []
    if not len(clfs):
        clfs = [LogisticRegression()]

    for idx, clf in enumerate(clfs):
        scores = cross_val_score(clf, X_train, y_train, cv=5, scoring=make_scorer(f1_score, average='weighted'))
        clf_scores += [np.mean(scores)]


    best_model = clfs[clf_scores.index(max(clf_scores))]
    return best_model

def tune_model(X_train, y_train, clfs=None, params=None, n_iter=10, grid_search=False):
    """
    Tunes different models with a few parameter combinations and returns a list of best performing models
    :param clfs:
    :param params:
    :return:
    """
    tuned_models = []

    if not (clfs and params):
        clfs = [
            LogisticRegression(), 
            RandomForestClassifier(n_estimators=20), 
            XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
               gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
               min_child_weight=1, missing=None,  n_estimators=80, nthread=-1,
               objective='multi:softmax', reg_lambda=1,
               scale_pos_weight=1, seed=0, silent=True, subsample=0.8)
        ]
        params = [
            {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [.001, .01, .5, 0.1, 0.5]},
            {"max_depth": [3, 5, 10, 12], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
            {'reg_lambda': [0.1, 0.5, 1.5], 'max_depth': [10, 20, 15], 'n_estimators': [100, 120, 150]
}
        ]
    for clf, param in zip(clfs, params):
        if grid_search:
            rscv = GridSearchCV(clf, param_grid=params)
        else:
            rscv = RandomizedSearchCV(clf, param_distributions=param, n_iter=n_iter)
        tuned_models += [rscv.fit(X_train, y_train)]

    return tuned_models

y = np.array(data_clean.interest_level)
X = np.array(data_clean.drop('interest_level', axis=1))
Xtrain, ytrain, Xtest, ytest = split_test_train(X, y)
Xtrain.shape, ytrain.shape, Xtest.shape, ytest.shape
clfs = tune_model(Xtrain, ytrain)

classifier = train_model(Xtrain, ytrain, clfs)

pred = classifier.predict(Xtest)

print(pd.DataFrame(confusion_matrix(ytest, pred, labels=[0,1,2])))
print(classification_report(y_true=ytest, y_pred=pred, target_names=_classes))
