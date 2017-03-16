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
        clfs = [LogisticRegression().fit(X_train, y_train)]

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
#             LogisticRegression(), 
            RandomForestClassifier(n_estimators=20), 
            XGBClassifier(base_score=0.5, colsample_bylevel=1, colsample_bytree=0.8,
               gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=10,
               min_child_weight=1, missing=None,  n_estimators=80, nthread=-1,
               objective='multi:softmax', reg_lambda=1,
               scale_pos_weight=1, seed=0, silent=True, subsample=0.8)
        ]
        params = [
#             {'solver': ['liblinear'], 'penalty': ['l1', 'l2'], 'C': [.001, .01, 0.5]},
            {"max_depth": [3, 5, 10, 12], "bootstrap": [True, False], "criterion": ["gini", "entropy"]},
            {'reg_lambda': [0.1, 0.5, 1.5], 'max_depth': [10, 20, 15], 'n_estimators': [100, 120, 150]
}
        ]
    for clf, param in zip(clfs, params):
        if grid_search:
            rscv = GridSearchCV(clf, param_grid=param)
        else:
            rscv = RandomizedSearchCV(clf, param_distributions=param, n_iter=n_iter)
        tuned_models += [rscv.fit(X_train, y_train)]

    return tuned_models



