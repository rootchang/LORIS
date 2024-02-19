###############################################################################################
#Aim: Machine learning model evaluation
#Description: To evaluate NSCLC-specific machine learning model performance using 2000-repeated 5-fold cross validation.
#
#Run command, e.g.: python 08_2.NSCLC_20Models_evaluation.py TMB 1
###############################################################################################


import time
import sys
import pandas as pd
from collections import Counter
import ast
from sklearn.gaussian_process import kernels
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer, matthews_corrcoef, balanced_accuracy_score
import utils2


if __name__ == "__main__":
    start_time = time.time()

    ############################################## 0. Parameters setting ##############################################
    MLM_list1=['TMB', 'RF6', 'RF16_NBT', 'DecisionTree', 'RandomForest'] # data scaling: None
    MLM_list2=['LLR6', 'LLR5noTMB', 'LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost',
               'LightGBM', 'SupportVectorMachineRadial','kNearestNeighbourhood','NeuralNetwork1','NeuralNetwork2',
               'NeuralNetwork3','NeuralNetwork4','GaussianProcess'] # StandardScaler
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
    dataset = 'Chowell'
    CPU_num = -1
    N_repeat_KFold_paramTune = 1
    N_repeat_KFold = 2000
    info_shown = 1
    Kfold = 5
    Kfold_list = RepeatedKFold(n_splits=Kfold, n_repeats=N_repeat_KFold,random_state=randomSeed)
    randomSearchNumber = 1

    phenoNA = 'Response'

    if MLM in ['LLR6', 'RF6']:
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    elif MLM == 'LLR5noTMB':
        featuresNA = ['PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    elif MLM == 'TMB':
        featuresNA = ['TMB']
    else:
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex',
                      'MSI', 'Stage', 'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI']
    xy_colNAs = featuresNA + [phenoNA]
    cat_features = []


    ################################################# 1. Data read in #################################################
    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    data_train1 = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    data_train2 = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    data_train = pd.concat([data_train1, data_train2], axis=0)
    data_train = data_train.loc[data_train['CancerType'] == 'NSCLC',]
    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    data_train['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_train['TMB']]
    data_train['Age'] = [c if c < Age_upper else Age_upper for c in data_train['Age']]
    data_train['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_train['NLR']]

    data_train = data_train[xy_colNAs].dropna()
    counter = Counter(data_train[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    print('  Dataset: ', dataset, '. ML: ', MLM, '. Scaler: ', SCALE)
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
    if MLM not in ['GaussianProcess','LLR6', 'TMB', 'LLR5noTMB', 'RF16_NBT']:
        HyperParam_fnIn = '../03.Results/16features/NSCLC/NSCLC_'+dataset+'_ModelParaSearchResult_' + MLM + \
                          '_Scaler(' + SCALE + ')_CV' + str(Kfold)+'Rep'+\
                          str(N_repeat_KFold_paramTune) + '_random' + str(randomSeed) + '.txt'
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
    elif MLM in ['LLR6', 'LLR5noTMB']:
        param_dict = {'solver': ['saga'], 'penalty': ['elasticnet'], 'max_iter': [100], 'l1_ratio': [1],
                      'class_weight': ['balanced'], 'C': [0.1]}
    elif MLM == 'TMB':
        param_dict = {'penalty': ['none']}
    elif MLM == 'GaussianProcess':
        param_dict = {'optimizer': ['fmin_l_bfgs_b'], 'n_restarts_optimizer': [0], 'max_iter_predict': [100],
                      'kernel': [1 * kernels.RBF(length_scale=1)]}
    elif MLM == 'RF16_NBT':
        param_dict = {'n_estimators': [1000], 'min_samples_split': [2], 'min_samples_leaf': [20], 'max_depth': [8]}
    ############################### 2. Optimal model hyperparameter combination search ################################
    MLM_equiv = MLM
    if MLM in ['LLR6', 'TMB', 'LLR5noTMB']:
        MLM_equiv = 'LogisticRegression'
    elif MLM in ['RF6', 'RF16_NBT']:
        MLM_equiv = 'RandomForest'
    search_cv = utils2.optimalHyperParaSearcher(MLM_equiv, SCALE, data_train, featuresNA, phenoNA,scoring_dict, \
        randomSeed, CPU_num, N_repeat_KFold, info_shown,Kfold,cat_features, randomSearchNumber, param_dict, True)

    results_df = pd.DataFrame(search_cv.cv_results_)
    if MLM in ['LLR6', 'LLR5noTMB', 'RF6', 'TMB']:
        model_eval_fn = '../03.Results/6features/NSCLC/NSCLC_'+dataset+'_ModelEvalResult_' + MLM + '_Scaler(' + SCALE +\
                        ')_CV' + str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    else:
        model_eval_fn = '../03.Results/16features/NSCLC/NSCLC_'+dataset+'_ModelEvalResult_' + MLM + '_Scaler(' + SCALE \
                        + ')_CV' +str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    results_df.to_csv(model_eval_fn, sep='\t')
    print('Model evaluation done! Time used: ',time.time()-start_time)
