###############################################################################################
#Aim: Machine learning model evaluation
#Description: To evaluate pan-cancer machine learning model performance using 2000-repeated 5-fold cross validation.
#
#Run command, e.g.: python 05_2.PanCancer_20Models_evaluation.py TMB 1
###############################################################################################


import time
import sys
import pandas as pd
from collections import Counter
import ast
from sklearn.gaussian_process import kernels
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score
import utils2


if __name__ == "__main__":
    start_time = time.time()

    ############################################## 0. Parameters setting ##############################################
    MLM_list1=['TMB', 'RF6', 'DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes',
               'GaussianNaiveBayes', 'BernoulliNaiveBayes', 'RF16_NBT'] # data scaling: None
    MLM_list2=['LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost', 'LightGBM',
               'SupportVectorMachineLinear','SupportVectorMachinePoly','SupportVectorMachineRadial',
               'kNearestNeighbourhood','DNN','NeuralNetwork1','NeuralNetwork2','NeuralNetwork3','NeuralNetwork4',
               'GaussianProcess','QuadraticDiscriminantAnalysis', 'LLR6', 'LR5noTMB'] # StandardScaler
    MLM = sys.argv[1]
    if MLM in MLM_list1:
        SCALE = 'None'
    elif MLM in MLM_list2:
        SCALE = 'StandardScaler'
    else:
        raise Exception('MLM not recognized!')
    try:
        randomSeed = int(sys.argv[2])
    except:
        randomSeed = 1
    CPU_num = -1
    N_repeat_KFold_paramTune = 1
    N_repeat_KFold = 2000
    info_shown = 1
    Kfold = 5 # 5
    Kfold_list = RepeatedKFold(n_splits=Kfold, n_repeats=N_repeat_KFold, random_state=randomSeed)
    randomSearchNumber = 1
    phenoNA = 'Response'
    rf16 = ["CancerType_grouped", "Albumin", "HED", "TMB", "FCNA", "BMI", "NLR", "Platelets", "HGB", "Stage", "Age", "Drug",
            "Chemo_before_IO", "HLA_LOH", "MSI", "Sex"]
    rf6 = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', "CancerType_grouped"]
    if MLM ==  'LLR6':
        featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    elif MLM ==  'LR5noTMB':
        featuresNA = ['Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    else:
        featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'FCNA', 'NLR', 'Age','Drug', 'Sex', 'MSI', 'Stage',
                      'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16'] ## all 16 features

    cat_features = []

    ################################################# 1. Data read in #################################################
    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    data_train = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    data_train['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_train['TMB']]
    data_train['Age'] = [c if c < Age_upper else Age_upper for c in data_train['Age']]
    data_train['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_train['NLR']]
    counter = Counter(data_train[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    print('  Number of all features: ', len(featuresNA), '\n  Their names: ', featuresNA)
    print('  Phenotype name: ', phenoNA)
    print('  Negative/Positive samples in training set: ', pos_weight)
    print('Data size: ', data_train.shape[0])

    scoring_dict = {"AUC": "roc_auc",
                    "PRAUC": "average_precision",
                    "Accuracy": "accuracy",
                    "F1": 'f1',
                    "Precison": "precision",
                    "Recall": "recall",
                    "MCC": make_scorer(matthews_corrcoef),
                    "Balanced Accuracy": make_scorer(balanced_accuracy_score),
                    }
    ############## read-in the dictionary of optimal parameter combination from file ############
    if MLM not in ['GaussianProcess','RF16_NBT','RF6', 'TMB', 'LR5noTMB', 'LLR6']:
        HyperParam_fnIn = '../03.Results/16features/PanCancer/ModelParaSearchResult_' + MLM + '_Scaler(' + SCALE +\
                          ')_CV' + str(Kfold) + 'Rep' + str(N_repeat_KFold_paramTune) + '_random' + \
                          str(randomSeed) + '.txt'
        paramDict_line_str = 'Best params on CV sets:  '
        for line in open(HyperParam_fnIn,'r').readlines():
            if line.startswith(paramDict_line_str):
                paramDict_str = line.strip().split(paramDict_line_str)[1]
                break
        param_dict = ast.literal_eval(paramDict_str)
        for c in param_dict:
            param_dict[c] = [param_dict[c]]
        if MLM == "HGBoost":
            param_dict['loss'] = ['log_loss']
    elif MLM == 'RF16_NBT':
        param_dict = {'n_estimators': [1000], 'min_samples_split': [2], 'min_samples_leaf': [20], 'max_depth': [8]}
    elif MLM == 'RF6':
        param_dict = {'n_estimators': [900], 'min_samples_split': [20], 'min_samples_leaf': [8], 'max_depth': [8]}
    elif MLM == 'TMB':
        param_dict = {'penalty': ['none']}
    elif MLM == 'GaussianProcess':
        param_dict = {'optimizer': [None], 'n_restarts_optimizer': [0], 'max_iter_predict': [100],
                      'kernel': [10 * kernels.RBF(length_scale=10)]}
    elif MLM == 'LR5noTMB':
        param_dict = {'C': [0.01], 'class_weight': ['balanced'], 'l1_ratio': [0.4], 'max_iter': [100],
                      'penalty': ['elasticnet'], 'solver': ['saga']}
    elif MLM == 'LLR6':
        param_dict = {'C': [0.1], 'class_weight': ['balanced'], 'l1_ratio': [1], 'max_iter': [100],
                      'penalty': ['elasticnet'], 'solver': ['saga']}
        ############################# 2. Optimal model hyperparameter combination search ##############################
    if MLM not in ['RF16_NBT','RF6', 'TMB']:
        MLM_temp = MLM
        if MLM in ['LLR6', 'LLR5noChemo', 'LR5noTMB', 'LLR5noCancer']:
            MLM_temp = 'LogisticRegression'
        search_cv = utils2.optimalHyperParaSearcher(MLM_temp, SCALE, data_train, featuresNA, phenoNA,scoring_dict, \
            randomSeed, CPU_num, N_repeat_KFold, info_shown,Kfold,cat_features, randomSearchNumber, param_dict, True)
    elif MLM == 'RF16_NBT':
        rf = RandomForestClassifier(random_state=randomSeed) # , n_jobs=CPU_num
        search_cv = RandomizedSearchCV(estimator=rf, param_distributions=param_dict, n_iter=randomSearchNumber,
                                       random_state=randomSeed, scoring=scoring_dict, refit='AUC',
                                       return_train_score=True, cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
        y = data_train[phenoNA]
        x = pd.DataFrame(data_train, columns=rf16)
        search_cv.fit(x, y)
    elif MLM == 'RF6':
        rf = RandomForestClassifier(random_state=randomSeed) # , n_jobs=CPU_num
        search_cv = RandomizedSearchCV(estimator=rf, param_distributions=param_dict, n_iter=randomSearchNumber,
                                       random_state=randomSeed, scoring=scoring_dict, refit='AUC',
                                       return_train_score=True, cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
        y = data_train[phenoNA]
        x = pd.DataFrame(data_train, columns=rf6)
        search_cv.fit(x, y)
    elif MLM == 'TMB':
        llr = linear_model.LogisticRegression()
        search_cv = RandomizedSearchCV(estimator=llr, param_distributions=param_dict, n_iter=randomSearchNumber,
                                       random_state=randomSeed, scoring=scoring_dict, refit='AUC',
                                       return_train_score=True, cv=Kfold_list, verbose=info_shown, n_jobs=CPU_num)
        y = data_train[phenoNA]
        x = pd.DataFrame(data_train, columns=['TMB'])
        search_cv.fit(x, y)
    results_df = pd.DataFrame(search_cv.cv_results_)
    if MLM in ['LLR6', 'LR5noTMB', 'RF6', 'TMB']:
        model_eval_fn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + \
                        str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    else:
        model_eval_fn = '../03.Results/16features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' +\
                    str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    results_df.to_csv(model_eval_fn, sep='\t')
    print('Model evaluation done! Time used: ',time.time()-start_time)
