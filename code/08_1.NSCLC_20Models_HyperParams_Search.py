###############################################################################################
#Aim: Hyper-parameter search
#Description: To search the optimal parameters for NSCLC-specific machine learning models.
#
#Run command, e.g.: python 08_1.NSCLC_20Models_HyperParams_Search.py DecisionTree 1
###############################################################################################


import time
import sys
import pandas as pd
from collections import Counter
import utils2


if __name__ == "__main__":
    start_time = time.time()

    ############################################## 0. Parameters setting ##############################################
    MLM_list1=['RF6', 'DecisionTree', 'RandomForest'] # data scaling: None
    MLM_list2=['LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'CatBoost', 'LightGBM',
               'SupportVectorMachineRadial','kNearestNeighbourhood','NeuralNetwork1','NeuralNetwork2','NeuralNetwork3',
               'NeuralNetwork4','GaussianProcess'] # StandardScaler
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
    N_repeat_KFold = 1
    info_shown = 1
    Kfold = 5
    randomSearchNumber = 10000

    phenoNA = 'Response'
    model_hyperParas_fn = '../03.Results/NSCLC_Chowell_ModelParaSearchResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(
        Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    if MLM not in ['RF6']:
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Systemic_therapy_history', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI',
                      'Stage', 'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI']
    else:
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age']
    dataALL_fn = '../02.Input/AllData.xlsx'
    data_train1 = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
    data_train2 = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
    data_train = pd.concat([data_train1,data_train2],axis=0)
    data_train = data_train.loc[data_train['CancerType']=='NSCLC',]

    if MLM == 'RF6':
        MLM = 'RandomForest'
    xy_colNAs = featuresNA + [phenoNA]
    data_train = data_train[xy_colNAs].dropna()
    cat_features = []

    ################################################# 1. Data read in #################################################
    model_hyperParas_fh = open(model_hyperParas_fn, 'w')
    print('Raw data processing ...', file=model_hyperParas_fh)

    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    data_train['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_train['TMB']]
    data_train['Age'] = [c if c < Age_upper else Age_upper for c in data_train['Age']]
    data_train['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_train['NLR']]
    counter = Counter(data_train[phenoNA])
    pos_weight = counter[0] / counter[1]
    print('  Number of all features: ', len(featuresNA), '\n  Their names: ', featuresNA, file=model_hyperParas_fh)
    print('  Phenotype name: ', phenoNA, file=model_hyperParas_fh)
    print('  Negative/Positive samples in training set: ', pos_weight, file=model_hyperParas_fh)
    print('Data size: ', data_train.shape[0], file=model_hyperParas_fh)

    scoring_dict = 'roc_auc'
    ############################### 2. Optimal model hyperparameter combination search ################################
    search_cv = utils2.optimalHyperParaSearcher(MLM, SCALE, data_train, featuresNA, phenoNA,scoring_dict, \
        randomSeed, CPU_num, N_repeat_KFold, info_shown,Kfold,cat_features, randomSearchNumber)
    print('Best params on CV sets: ', search_cv.best_params_, file=model_hyperParas_fh)
    print('Best score on CV sets: ', search_cv.best_score_, file=model_hyperParas_fh)
    print('Hyperparameter screening done! Time used: ',time.time()-start_time, file=model_hyperParas_fh)
