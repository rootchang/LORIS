#############################################################################################
# Random search of hyperparameters for a collection of ML models
#############################################################################################

import copy
import sys
import pandas as pd
import numpy as np
import itertools

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier, kernels
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import Counter


def AUC_calculator(y, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auc, thresholds[ind_max]


def dataScaler(data, featuresNA, numeric_featuresNA, scaler_type):
    data_scaled = copy.deepcopy(data)
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise Exception('Unrecognized scaler type of %s! Only "sd" and "mM" are accepted.' % scaler_type)
    for feature in numeric_featuresNA:
        data_scaled[feature] = scaler.fit_transform(data[[feature]])
    x = pd.DataFrame(data_scaled, columns=featuresNA)
    return x


def DecisionTree_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                          scoring_dict, searchN, params=[], returnScore=False):
    dt = DecisionTreeClassifier(criterion="gini", class_weight='balanced',random_state=randomSeed)
    if not params:
        params = {'splitter': ['best', 'random'],
                  'max_features': list(np.arange(0.1, 0.91, 0.1)), # 11
                  'max_depth': list(range(3, 11)), # 11
                  'min_samples_leaf': list(range(2, 31, 2)),
                  'min_samples_split': list(range(2, 31, 2)),
                  'ccp_alpha': [0, 0.5, 1, 10, 100],
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=dt, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def RandomForest_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                          scoring_dict, searchN, params=[], returnScore=False):
    rf = RandomForestClassifier(bootstrap=True, criterion="gini", class_weight='balanced_subsample',
                                random_state=randomSeed, n_jobs=CPU_num)
    if not params:
        params = {'n_estimators': list(np.arange(200, 2100, 200)),
                  'max_features': list(np.arange(0.1, 0.91, 0.1)),
                  'max_depth': list(range(3, 11)), # 11
                  'min_samples_leaf': list(range(2, 31, 2)),
                  'min_samples_split': list(range(2, 31, 2)),
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=rf, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def GBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                    searchN, params=[], returnScore=False):
    gb_model = GradientBoostingClassifier(random_state=randomSeed)
    if not params:
        params = {
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            'n_estimators': list(np.arange(200, 2100, 200)),
            'min_samples_split': list(range(2, 31, 2)),
            'min_samples_leaf': list(range(2, 31, 2)),
            'max_depth': list(np.arange(3, 11, 1)),
            'max_features': list(np.arange(0.1, 0.91, 0.1)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=gb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def AdaBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                      cat_features, searchN, params=[], returnScore=False):
    base_estim = DecisionTreeClassifier(max_depth=1)
    ABM = AdaBoostClassifier(base_estimator=base_estim, random_state=randomSeed)
    if not params:
        params = {'n_estimators': list(np.arange(200, 2001, 200)),
                  'learning_rate': [0.01, 0.05, 0.03, 0.1, 0.3, 0.5, 1],
                  'algorithm': ['SAMME', 'SAMME.R']
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=ABM, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def HGBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                     info_shown, scoring_dict, cat_features, searchN, params=[], returnScore=False):
    featuresNA = list(x_train.columns)
    cat_features_index = list(map(lambda x: featuresNA.index(x), cat_features))
    hgb_model = HistGradientBoostingClassifier(categorical_features=cat_features_index, random_state=randomSeed)
    if not params:
        params = {
            'loss': ['auto'],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5],
            'max_iter': list(np.arange(200, 2001, 200)),
            'min_samples_leaf': list(range(2, 31, 2)),
            'max_depth': list(np.arange(3, 11, 1)),
            'l2_regularization': [0] + list(10 ** np.arange(-4, 2.1, 1))
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=hgb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def XGBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                     searchN, params=[], returnScore=False):
    xgb_model = XGBClassifier(objective='binary:logistic', scale_pos_weight=pos_weight, random_state=randomSeed)
    if not params:
        params = {
            'min_child_weight': [1] + list(range(2, 31, 2)),
            'gamma': [0] + list(10**np.arange(-2,0.1,1)),
            'subsample': [0.5, 0.8, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'max_depth': list(np.arange(3, 11, 1)),
            'n_estimators': [100]+list(np.arange(200, 1100, 200)),
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.3, 0.5]
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN, 1000)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=xgb_model, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def CatBoost_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown,
                      scoring_dict, cat_features, searchN, params=[], returnScore=False):
    CBM = CatBoostClassifier(class_weights=[1, pos_weight], cat_features=cat_features,
                             logging_level='Silent', random_seed=randomSeed) # , thread_count=-1
    if not params:
        params = {'depth': list(np.arange(3, 11, 1)),
                  'learning_rate': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1],
                  'iterations': [100]+list(np.arange(200, 1100, 200)),
                  'subsample': [0.5, 0.8, 1],
                  'reg_lambda': [0, 1, 2, 3, 5, 10],
                  'min_data_in_leaf': list(range(2, 31, 2)),
                  'random_strength': [0, 1, 2, 3, 5, 10]
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=CBM, param_distributions=params,
                                   n_iter=real_searchN, random_state=randomSeed, cv=Kfold_list, verbose=info_shown,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def LightGBM_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,
                      cat_features, searchN, params=[], returnScore=False):
    featuresNA = list(x_train.columns)
    cat_features_index = list(map(lambda x: featuresNA.index(x), cat_features))
    lgb_estimator = lgb.LGBMClassifier(scale_pos_weight=pos_weight, cat_feature=cat_features_index,
                                       random_state=randomSeed, n_jobs=CPU_num)
    if not params:
        params = {
            'learning_rate': [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1, 0.3],
            'max_depth': list(np.arange(3, 11, 1)),
            'n_estimators': list(np.arange(200, 2100, 200)),
            'num_leaves': list(np.arange(10, 101, 10)),
            'colsample_bytree': [0.2,0.4,0.6,0.8,1],
            'min_data_in_leaf': list(np.arange(2, 31, 2))
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=lgb_estimator, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, verbose=info_shown, scoring=scoring_dict,
                                   refit = 'AUC', return_train_score=True,random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train, y_train)
    return search_cv


def ElasticNet_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                        info_shown, scoring_dict, searchN, params=[], returnScore=False):
    eNet = linear_model.ElasticNet(random_state=randomSeed)
    if not params:
        params = {"max_iter": list(np.arange(100, 5100, 200)),
                  "alpha": list(10 ** np.arange(-4, 2.1, 0.6)),
                  "l1_ratio": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  "fit_intercept": [True, False],
                  "tol": list(10 ** np.arange(-5, -0.9, 0.4)),
                  "selection": ['cyclic', 'random']
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(eNet, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,
                                   n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    return search_cv


def LogisticRegression_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                info_shown, scoring_dict, searchN, params=[], returnScore=False):
    LR = linear_model.LogisticRegression(random_state=randomSeed)
    if not params:
        params = {'solver': ['saga'],
                  'l1_ratio': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
                  'max_iter': range(100,1100,100),
                  'penalty': ['elasticnet'],
                  'C': list(10 ** np.arange(-3, 3.01, 1)),
                  'class_weight': ['balanced'],
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(LR, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,
                                   n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    return search_cv


def SupportVectorMachineRadial_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                  info_shown, scoring_dict, searchN, params=[], returnScore=False):
    SVM = SVC(probability=True)
    if not params:
        params = {'C': list(10 ** (np.arange(-5, 3.1, 0.5))),
                  'gamma': ['scale', 'auto'] + list(10 ** (np.arange(-4, 2.1, 0.5))),
                  'kernel': ['rbf'],
                  'max_iter': [-1,100,1000],
                  'tol': list(10 ** (np.arange(-5, -0.9, 0.5))),
                  'class_weight': [None, 'balanced']
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(SVM, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,
                                   n_jobs=CPU_num, cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    return search_cv


def kNearestNeighbourhood_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                                   info_shown, scoring_dict, searchN, params=[], returnScore=False):
    KNN = KNeighborsClassifier(n_jobs=CPU_num)
    if not params:
        params = {"n_neighbors": list(range(2, 61, 2)),
                  "weights": ['uniform', 'distance'],
                  "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
                  "leaf_size": list(range(2, 31, 2)),
                  "p": list(np.arange(1, 11, 1))
                  }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(KNN, param_distributions=params, n_iter=real_searchN, 
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,n_jobs=CPU_num,
                                   cv=Kfold_list, verbose=info_shown)
    search_cv.fit(x_train, y_train)
    return search_cv


def NeuralNetwork1_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'solver': ['sgd','lbfgs', 'adam'],
            'learning_rate': ["constant", "invscaling", "adaptive"],
            'max_iter': [100, 200, 500, 1000],
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 41), repeat=1)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],
            'alpha': list(np.logspace(-6, -1, num=6)),
            'early_stopping': [False, True]
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, verbose=info_shown, random_state=randomSeed,
                                   n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv

def NeuralNetwork2_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'max_iter': [100, 200, 1000],
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=2)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],
            'alpha': list(np.logspace(-6, -1, num=6)),
            'early_stopping': [False, True]
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, verbose=info_shown, random_state=randomSeed,
                                   n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv


def NeuralNetwork3_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,
                           info_shown, scoring_dict, searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=3)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],
            'alpha': list(np.logspace(-6, -1, num=6))
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, verbose=info_shown, random_state=randomSeed,
                                   n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv


def NeuralNetwork4_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict,
                            searchN, params=[], returnScore=False):
    MLP = MLPClassifier(random_state=randomSeed)
    if not params:
        params = {
            'hidden_layer_sizes': [x for x in itertools.product(range(2, 21), repeat=4)],
            'activation': ['logistic', 'tanh', 'relu', 'identity'],
            'alpha': list(np.logspace(-6, -1, num=6))
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=MLP, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore,verbose=info_shown, random_state=randomSeed,
                                   n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv


def GaussianProcess_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict,
                             searchN, params=[], returnScore=False):
    GPC = GaussianProcessClassifier(random_state=randomSeed,n_jobs = -1)
    if not params:
        params = {
            'kernel': [None,1.0 * kernels.RBF(1.0),0.1 * kernels.RBF(0.1),10 * kernels.RBF(10)],
            'optimizer': ['fmin_l_bfgs_b',None],
            'max_iter_predict': [100,500,1000],
            'n_restarts_optimizer': [0,5,10,15,20,25,30]
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=GPC, param_distributions=params,
                                   n_iter=real_searchN, cv=Kfold_list, scoring=scoring_dict, refit = 'AUC',
                                   return_train_score=returnScore, verbose=info_shown, random_state=randomSeed,
                                   n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv


def QuadraticDiscriminantAnalysis_searcher(x_train, y_train, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown,
                                           scoring_dict, searchN, params=[], returnScore=False):
    QDA = QuadraticDiscriminantAnalysis()
    if not params:
        params = {
            'reg_param': [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],
            'store_covariance': [True,False],
            'tol': list(10 ** np.arange(-6, -0.9, 0.5)),
        }
        param_combinations = np.product([len(params[c]) for c in params])
    else:
        param_combinations = len(params)
    real_searchN = min(param_combinations, searchN)
    print('Total number of models for searching: ', real_searchN)
    search_cv = RandomizedSearchCV(estimator=QDA, param_distributions=params,n_iter=real_searchN, cv=Kfold_list,
                                   scoring=scoring_dict, refit = 'AUC', return_train_score=returnScore,
                                    verbose=info_shown, random_state=randomSeed, n_jobs=CPU_num)
    search_cv.fit(x_train.values, y_train.values)
    return search_cv


def optimalHyperParaSearcher(MLM, SCALE, data, featuresNA, phenoNA, scoring_dict, randomSeed, CPU_num,N_repeat_KFold,
                             info_shown, Kfold, cat_features, searchN, params_dict_list=[], returnScore=False):
    y = data[phenoNA]
    counter = Counter(y)
    pos_weight = counter[0] / counter[1]

    if SCALE == 'None':
        x = pd.DataFrame(data, columns=featuresNA)
    else:
        numeric_featuresNA = list(set(featuresNA) - set(cat_features))
        if SCALE == 'StandardScaler':
            x = dataScaler(data, featuresNA, numeric_featuresNA, 'StandardScaler')
        elif SCALE == 'MinMax':
            x = dataScaler(data, featuresNA, numeric_featuresNA, 'MinMax')
        else:
            raise Exception('Unrecognized SCALE of %s! Only "None", "StandardScaler" and "MinMax" are supported.'% SCALE)

    if Kfold > 1.5:
        Kfold_list = RepeatedKFold(n_splits=Kfold, n_repeats=N_repeat_KFold, random_state=randomSeed) # RepeatedStratifiedKFold
    else:
        Kfold_list = [(slice(None), slice(None))]

    if MLM == 'DecisionTree':
        search_cv = DecisionTree_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'RandomForest':
        search_cv = RandomForest_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'GBoost':
        search_cv = GBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'AdaBoost':
        search_cv = AdaBoost_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'HGBoost':
        search_cv = HGBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'XGBoost':
        search_cv = XGBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'CatBoost':
        search_cv = CatBoost_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'LightGBM':
        search_cv = LightGBM_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,cat_features, searchN, params_dict_list, returnScore)
    elif MLM == 'ElasticNet':
        search_cv = ElasticNet_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'LogisticRegression':
        search_cv = LogisticRegression_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list, info_shown,scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'SupportVectorMachineRadial':
        search_cv = SupportVectorMachineRadial_searcher(x, y, pos_weight,randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'kNearestNeighbourhood':
        search_cv = kNearestNeighbourhood_searcher(x, y, pos_weight, randomSeed, CPU_num, Kfold_list,info_shown, scoring_dict, searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork1':
        search_cv = NeuralNetwork1_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork2':
        search_cv = NeuralNetwork2_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork3':
        search_cv = NeuralNetwork3_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'NeuralNetwork4':
        search_cv = NeuralNetwork4_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'GaussianProcess':
        search_cv = GaussianProcess_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    elif MLM == 'QuadraticDiscriminantAnalysis':
        search_cv = QuadraticDiscriminantAnalysis_searcher(x, y, pos_weight, randomSeed,CPU_num, Kfold_list, info_shown, scoring_dict,searchN, params_dict_list, returnScore)
    else:
        raise Exception('Unrecognized machine learning algorithm MLM of %s!' % MLM)
    return search_cv
