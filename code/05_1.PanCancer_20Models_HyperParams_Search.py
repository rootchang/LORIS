###############################################################################################
#Aim: Hyper-parameter search
#Description: To search the optimal parameters for pan-cancer machine learning models.
#
#Run command, e.g.: python 05_1.PanCancer_20Models_HyperParams_Search.py DecisionTree 1
###############################################################################################


import time
import sys
import pandas as pd
from collections import Counter
import utils2


if __name__ == "__main__":
    start_time = time.time()

    ############################################## 0. Parameters setting ##############################################
    MLM_list1=['DecisionTree', 'RandomForest'] # data scaling: None
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
    CPU_num = 8
    N_repeat_KFold = 1
    info_shown = 1
    Kfold = 5
    randomSearchNumber = 10000
    model_hyperParas_fn = '../03.Results/16features/PanCancer/ModelParaSearchResult_' + MLM + '_Scaler(' + \
                          SCALE + ')_CV' + str(Kfold)+'Rep'+str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    model_hyperParas_fh = open(model_hyperParas_fn,'w')
    phenoNA = 'Response'
    featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'FCNA', 'NLR', 'Age','Drug', 'Sex', 'MSI', 'Stage',
                  'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'CancerType1',
                  'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                  'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                  'CancerType14', 'CancerType15', 'CancerType16'] ## all 16 features
    xy_colNAs = featuresNA + [phenoNA]
    cat_features = []

    ################################################# 1. Data read in #################################################
    print('Raw data processing ...', file=model_hyperParas_fh)
    dataALL_fn = '../02.Input/AllData.xlsx'
    data_train = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
    # Data truncation
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    try:
        data_train['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_train['TMB']]
    except:
        1
    data_train['Age'] = [c if c < Age_upper else Age_upper for c in data_train['Age']]
    data_train['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_train['NLR']]
    counter = Counter(data_train[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
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
