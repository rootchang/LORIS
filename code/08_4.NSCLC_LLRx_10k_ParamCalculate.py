###############################################################################################
#Aim: coefs. and intercepts for the LLR models
#Description: To determine the coefs. and intercepts of the
#             1) NSCLC-specific LLR6 model
#             2) NSCLC-specific LLR5noChemo model
#             3) NSCLC-specific LLR2 model
#             with 10k-repeat train-test splitting (80%:20%).
#
#Run command, e.g.: python 08_4.NSCLC_LLRx_10k_ParamCalculate.py LLR6
###############################################################################################


import sys
import time
import pandas as pd
import numpy as np
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import copy
from scipy import stats


if __name__ == "__main__":
    start_time = time.time()

    CPU_num = -1
    randomSeed = 1
    resampleNUM = 10000
    train_size = 0.8

    phenoNA = 'Response'
    LLRmodelNA = sys.argv[1]  # 'LLR6'   'LLR5noChemo'  'LLR2'
    if LLRmodelNA == 'LLR6':
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    elif LLRmodelNA == 'LLR5noChemo':
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Albumin', 'NLR', 'Age']
    elif LLRmodelNA == 'LLR2':
        featuresNA = ['TMB', 'PDL1_TPS(%)']
    xy_colNAs = featuresNA + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowell_Train0 = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    dataChowell_Train1 = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)

    dataChowell_Train0 = pd.concat([dataChowell_Train0,dataChowell_Train1],axis=0)

    dataChowell_Train0 = dataChowell_Train0.loc[dataChowell_Train0['CancerType']=='NSCLC',]
    dataChowell_Train0 = dataChowell_Train0[xy_colNAs].dropna(axis=0)
    dataChowell_Train = copy.deepcopy(dataChowell_Train0)

    # truncate extreme values of features
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    try:
        dataChowell_Train['TMB'] = [c if c < TMB_upper else TMB_upper for c in dataChowell_Train0['TMB']]
    except:
        1
    try:
        dataChowell_Train['Age'] = [c if c < Age_upper else Age_upper for c in dataChowell_Train0['Age']]
    except:
        1
    try:
        dataChowell_Train['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataChowell_Train0['NLR']]
    except:
        1
    print('Patient number (training): ', dataChowell_Train0.shape[0])
    counter = Counter(dataChowell_Train0[phenoNA])  # count examples in each class
    pos_weight = counter[0] / counter[1]  # estimate scale_pos_weight value
    print('  Phenotype name: ', phenoNA)
    print('  Negative/Positive samples in training set: ', pos_weight)

    ############## 10000-replicate random data splitting for model training and evaluation ############
    LR_params10000 = [[], [], [], [], []]  # norm_mean, norm_std, coefs, interc
    param_dict_LR6 = {'penalty': 'l1', 'C': 0.1, 'class_weight': 'balanced', 'solver': 'saga', 'random_state': randomSeed}

    test_size = 1 - train_size
    AUC_score_dict = {}
    for resampling_i in range(resampleNUM):
        data_train, data_test = train_test_split(dataChowell_Train, test_size=test_size, random_state=resampling_i*randomSeed,
                                                 stratify=None)  # stratify=None
        y_train = data_train[phenoNA]
        y_test = data_test[phenoNA]
        x_train6LR = pd.DataFrame(data_train, columns=featuresNA)
        x_test6LR = pd.DataFrame(data_test, columns=featuresNA)

        scaler_sd = StandardScaler()  # StandardScaler()
        x_train6LR = scaler_sd.fit_transform(x_train6LR)
        LR_params10000[0].append(list(scaler_sd.mean_))
        LR_params10000[1].append(list(scaler_sd.scale_))
        x_test6LR = scaler_sd.transform(x_test6LR)

        ############# LASSO Logistic Regression model #############
        clf = linear_model.LogisticRegression(**param_dict_LR6).fit(x_train6LR, y_train)
        LR_params10000[2].append(list(clf.coef_[0]))
        LR_params10000[3].append(list(clf.intercept_))

        predictions = clf.predict(x_train6LR)
        params = np.append(clf.intercept_, clf.coef_)
        newX = np.append(np.ones((len(x_train6LR), 1)), x_train6LR, axis=1)
        MSE = (sum((y_train - predictions) ** 2)) / (len(newX) - len(newX[0]))
        var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
        sd_b = np.sqrt(var_b)
        ts_b = params / sd_b
        p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
        LR_params10000[4].append(p_values[1:] + [p_values[0]])

    fnOut = open('../../03.Results/16features/NSCLC/NSCLC_'+LLRmodelNA+'_10k_ParamCalculate.txt', 'w', buffering=1)
    for i in range(5):
        LR_params10000[i] = list(zip(*LR_params10000[i]))
        LR_params10000[i] = [np.mean(c) for c in LR_params10000[i]]
    print('coef     : ', [round(c,4) for c in LR_params10000[2]])
    print('intercept: ', [round(c,4) for c in LR_params10000[3]])
    print('p_val: ', [round(c, 4) for c in LR_params10000[4]])
    fnOut.write('LLR_mean\t' + '\t'.join([str(c) for c in LR_params10000[0]]) + '\n')
    fnOut.write('LLR_scale\t' + '\t'.join([str(c) for c in LR_params10000[1]]) + '\n')
    fnOut.write('LLR_coef\t' + '\t'.join([str(c) for c in LR_params10000[2]]) + '\n')
    fnOut.write('LLR_intercept\t' + '\t'.join([str(c) for c in LR_params10000[3]]) + '\n')
    fnOut.write('LLR_pval\t' + '\t'.join([str(c) for c in LR_params10000[4]]) + '\n')
    fnOut.close()

    print('All done! Time used: ', time.time() - start_time)