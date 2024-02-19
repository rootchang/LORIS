###############################################################################################
#Aim: NSCLC-specific LLR6 vs. LLR5 comparison
#Description: AUC comparison between NSCLC-specific LLR6 vs. LLR5 on training and multiple test sets.
#             (Extended Data Fig. 9b).
#Run command: python 09_5.NSCLC_LLR6_vs_LLR5noChemo_ROC_AUC.py
###############################################################################################


import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from scipy.stats import norm
from sklearn.utils import resample

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")


def delong_test(y_true, y_pred1, y_pred2):
    n1 = len(y_pred1)
    n2 = len(y_pred2)
    # Compute AUCs
    auc1 = roc_auc_score(y_true, y_pred1)
    auc2 = roc_auc_score(y_true, y_pred2)
    # Compute Z statistic
    Q1 = auc1 * (1 - auc1)
    Q2 = auc2 * (1 - auc2)
    var = (Q1 / n1) + (Q2 / n2)
    Z_statistic = (auc1 - auc2) / np.sqrt(var)
    # Compute p-value
    p_value = 2.0 * (1 - norm.cdf(abs(Z_statistic)))
    return p_value


def AUC_with95CI_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    n_bootstrap = 1000
    auc_values = []
    for _ in range(n_bootstrap):
        y_true_bootstrap, y_scores_bootstrap = resample(y, y_pred, replace=True)
        try:
            AUC_temp = roc_auc_score(y_true_bootstrap, y_scores_bootstrap)
        except:
            continue
        auc_values.append(AUC_temp)
    lower_auroc = np.percentile(auc_values, 2.5)
    upper_auroc = np.percentile(auc_values, 97.5)
    return auroc, threshold[ind_max], lower_auroc, upper_auroc


if __name__ == "__main__":

    ########################## Read in data ##########################
    phenoNA = 'Response'
    featuresNA_LLR6 = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    featuresNA_LLR5 = ['TMB', 'PDL1_TPS(%)', 'Albumin', 'NLR', 'Age']
    xy_colNAs = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age'] + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    dataChowell = pd.concat([dataChowellTrain,dataChowellTest],axis=0)

    dataMorris_new = pd.read_excel(dataALL_fn, sheet_name='Morris_new', index_col=0)
    dataLee = pd.read_excel(dataALL_fn, sheet_name='Lee_NSCLC', index_col=0)

    dataVanguri = pd.read_excel(dataALL_fn, sheet_name='Vanguri_NSCLC_all', index_col=0)
    dataRavi = pd.read_excel(dataALL_fn, sheet_name='Ravi_NSCLC', index_col=0)
    dataRavi['Albumin'] = 3.8  # impute values for the LLR6 model
    dataRavi['NLR'] = 6.9 # impute values for the LLR6 model

    dataALL = [dataChowell, dataMorris_new, dataLee, dataVanguri, dataRavi]

    for i in range(len(dataALL)):
        dataALL[i] = dataALL[i].loc[dataALL[i]['CancerType']=='NSCLC',]
        dataALL[i] = dataALL[i][xy_colNAs].astype(float)
        dataALL[i] = dataALL[i].dropna(axis=0)

    # truncate TMB
    TMB_upper = 50
    try:
        for i in range(len(dataALL)):
            dataALL[i]['TMB'] = [c if c<TMB_upper else TMB_upper for c in dataALL[i]['TMB']]
    except:
        1
    # truncate Age
    Age_upper = 85
    try:
        for i in range(len(dataALL)):
            dataALL[i]['Age'] = [c if c < Age_upper else Age_upper for c in dataALL[i]['Age']]
    except:
        1
    # truncate NLR
    NLR_upper = 25
    try:
        for i in range(len(dataALL)):
            dataALL[i]['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataALL[i]['NLR']]
    except:
        1

    x_test_LLR6_list = []
    x_test_LLR5_list = []
    y_test_list = []
    for c in dataALL:
        x_test_LLR6_list.append(pd.DataFrame(c, columns=featuresNA_LLR6))
        x_test_LLR5_list.append(pd.DataFrame(c, columns=featuresNA_LLR5))
        y_test_list.append(c[phenoNA])


    y_LLR6pred_test_list = []
    y_LLR5pred_test_list = []
    ###################### test LLR6 model performance ######################
    fnIn = '../03.Results/16features/NSCLC/NSCLC_LLR6_10k_ParamCalculate.txt'
    params_data = open(fnIn,'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val
    x_test_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_LLR6_list[0][featuresNA_LLR6])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_LLR6_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA_LLR6])))
    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        y_LLR6pred_test_list.append(y_pred_test)

    ###################### test LLR5 model performance ######################
    fnIn = '../03.Results/16features/NSCLC/NSCLC_LLR5noChemo_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val
    x_test_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_LLR5_list[0][featuresNA_LLR5])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_LLR5_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA_LLR5])))
    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        y_LLR5pred_test_list.append(y_pred_test)

    ################ Test AUC difference p value between models ######
    pval_list = []
    for i in range(len(y_LLR6pred_test_list)):
        p1 = delong_test(y_test_list[i], y_LLR6pred_test_list[i], y_LLR5pred_test_list[i])
        pval_list.append(p1)
        print('Dataset %d: LLR6 vs LLR5 p-val: %g' % (i + 1, p1))

    ############################## save source data for figure ##############################
    dataset_list = []
    true_label_list = []
    LLR6_pred_list = []
    LLR5_pred_list = []
    dataset_unique = ["Chowell et al.","MSK1","Shim et al.","Vanguri et al.","Ravi et al."]
    for i in range(5):
        LLR6_pred_list.extend(y_LLR6pred_test_list[i])
        LLR5_pred_list.extend(y_LLR5pred_test_list[i])
        true_label_list.extend(y_test_list[i])
        dataset_list.extend([dataset_unique[i]]*len(y_test_list[i]))
    df = pd.DataFrame({"Dataset":dataset_list, "True_label":true_label_list, "LLR6_score":LLR6_pred_list, "LLR5_score":LLR5_pred_list})
    df.to_csv('../03.Results/source_data_efig10b.csv', index=False)


    ############################## Plot ROC curves ##############################
    textSize = 8
    output_fig1 = '../03.Results/LLR6_LLR5_compare_NSCLC.pdf'
    ax1 = [0] * 6
    fig1, ((ax1[0], ax1[1], ax1[2]), (ax1[3], ax1[4], ax1[5])) = plt.subplots(2, 3, figsize=(6.5, 3.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.5)
    fig1.delaxes(ax1[5])

    for i in range(5):
        y_true = y_test_list[i]
        ###### LLR6 model
        y_pred = y_LLR6pred_test_list[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        if not i:
            ax1[i].plot(fpr, tpr, color=palette[0], linestyle='-',
                        label='LLR6 AUC: %.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
        else:
            ax1[i].plot(fpr, tpr, color=palette[0], linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
        ###### LLR5 model
        y_pred = y_LLR5pred_test_list[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC, _, AUC_05, AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        if not i:
            ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-',
                        label='LLR5 AUC: %.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
            ax1[i].legend(frameon=False, loc=(0.1, 0), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
        else:
            ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
            ax1[i].legend(frameon=False, loc=(0.4, 0), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
        ax1[i].text(0.05, 0.9, 'p = %.2f' % pval_list[i])

        ax1[i].set_xlim([-0.02, 1.02])
        ax1[i].set_ylim([-0.02, 1.02])
        ax1[i].set_yticks([0,0.5,1])
        ax1[i].set_xticks([0,0.5,1])
        if i > 0 and i!=3:
            ax1[i].set_yticklabels([])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1)
    plt.close()
