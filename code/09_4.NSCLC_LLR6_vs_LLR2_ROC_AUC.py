###############################################################################################
#Aim: NSCLC-specific LLR6 vs. LLR2 comparison
#Description: AUC comparison between NSCLC-specific LLR6 vs. LLR2 on training and multiple test sets.
#             (Extended Data Fig. 6d)
#Run command: python 09_4.NSCLC_LLR6_vs_LLR2_ROC_AUC.py
###############################################################################################


import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.utils import resample
from scipy.stats import norm

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
    start_time = time.time()

    ########################## Read in data ##########################
    phenoNA = 'Response'
    cutoff_value_LLR6_NSCLC = 0.44
    featuresNA_LLR6 = ['TMB', 'PDL1_TPS(%)', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age']
    featuresNA_LLR2 = ['TMB', 'PDL1_TPS(%)']
    xy_colNAs = ['TMB', 'PDL1_TPS(%)', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age'] + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/AllData.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
    dataChowell = pd.concat([dataChowellTrain,dataChowellTest],axis=0)

    dataMSK1 = pd.read_excel(dataALL_fn, sheet_name='MSK1', index_col=0)
    dataLee = pd.read_excel(dataALL_fn, sheet_name='Shim_NSCLC', index_col=0)

    dataVanguri = pd.read_excel(dataALL_fn, sheet_name='Vanguri_NSCLC', index_col=0)
    dataRavi = pd.read_excel(dataALL_fn, sheet_name='Ravi_NSCLC', index_col=0)

    dataALL = [dataChowell, dataMSK1, dataLee, dataVanguri, dataRavi]

    for i in range(len(dataALL)):
        dataALL[i] = dataALL[i].loc[dataALL[i]['CancerType']=='NSCLC',]
        dataALL[i] = dataALL[i][xy_colNAs].astype(float)
        dataALL[i] = dataALL[i].dropna(axis=0)

    # truncate TMB
    TMB_upper = 50
    for i in range(len(dataALL)):
        dataALL[i]['TMB'] = [c if c<TMB_upper else TMB_upper for c in dataALL[i]['TMB']]
    # truncate Age
    Age_upper = 85
    for i in range(len(dataALL)):
        dataALL[i]['Age'] = [c if c < Age_upper else Age_upper for c in dataALL[i]['Age']]
    # truncate NLR
    NLR_upper = 25
    for i in range(len(dataALL)):
        dataALL[i]['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataALL[i]['NLR']]

    x_LLR6_test_list = []
    x_LLR2_test_list = []
    y_test_list = []
    for c in dataALL:
        x_LLR6_test_list.append(pd.DataFrame(c, columns=featuresNA_LLR6))
        x_LLR2_test_list.append(pd.DataFrame(c, columns=featuresNA_LLR2))
        y_test_list.append(c[phenoNA])


    ###################### Read in LLR6_NSCLC and LLR6_PanCancer model params ######################
    fnIn = '../03.Results/16features/NSCLC/NSCLC_LLR6_10k_ParamCalculate.txt'
    params_data = open(fnIn,'r').readlines()
    params_dict_LLR6 = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict_LLR6[param_name] = params_val

    fnIn = '../03.Results/16features/NSCLC/NSCLC_LLR2_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict_LLR2 = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict_LLR2[param_name] = params_val

    ########################## test LLR6_NSCLC and LLR6_PanCancer model performance ##########################
    y_pred_LLR6 = []
    x_test_LLR6_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_LLR6_test_list[0][featuresNA_LLR6])
    scaler_sd.mean_ = np.array(params_dict_LLR6['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict_LLR6['LLR_scale'])
    for c in x_LLR6_test_list:
        x_test_LLR6_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA_LLR6])))
    clf = linear_model.LogisticRegression().fit(x_test_LLR6_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict_LLR6['LLR_coef']])
    clf.intercept_ = np.array(params_dict_LLR6['LLR_intercept'])
    for i in range(len(x_test_LLR6_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_LLR6_scaled_list[i])[:, 1]
        y_pred_LLR6.append(y_pred_test)

    y_pred_LLR2 = []
    x_test_LLR2_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_LLR2_test_list[0][featuresNA_LLR2])
    scaler_sd.mean_ = np.array(params_dict_LLR2['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict_LLR2['LLR_scale'])
    for c in x_LLR2_test_list:
        x_test_LLR2_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA_LLR2])))
    clf = linear_model.LogisticRegression().fit(x_test_LLR2_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict_LLR2['LLR_coef']])
    clf.intercept_ = np.array(params_dict_LLR2['LLR_intercept'])
    for i in range(len(x_test_LLR2_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_LLR2_scaled_list[i])[:, 1]
        y_pred_LLR2.append(y_pred_test)


    ################ Test AUC difference p value between LLR6 and TMB models ######
    for i in range(len(y_pred_LLR2)):
        pval1 = delong_test(y_test_list[i], y_pred_LLR2[i], y_pred_LLR6[i]) #
        print('Dataset %d: NSCLC LLR6 vs LLR2 p-val: %.1g.'%(i+1,pval1))


    ############################## Plot ##############################
    textSize = 8

    ############# Plot ROC curves ##############
    output_fig1 = '../03.Results/NSCLC_LLR6_LLR2_AUC_compare.pdf'
    ax1 = [0] * 6
    fig1, ((ax1[0], ax1[1], ax1[2]), (ax1[3], ax1[4], ax1[5])) = plt.subplots(2, 3, figsize=(6.5, 3.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.5)
    fig1.delaxes(ax1[5])

    for i in range(5):
        y_true = y_test_list[i]
        ###### NSCLC specific LLR6 model
        y_pred = y_pred_LLR6[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        specificity_sensitivity_sum = tpr + (1 - fpr)
        ind_max = np.argmax(specificity_sensitivity_sum)
        if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
            ind_max = 1
        opt_cutoff = thresholds[ind_max]
        AUC, _, AUC_05, AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='LLR6 AUC: %.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        else:
            ax1[i].plot(fpr, tpr, color=palette[0], linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
        ###### Pancancer LLR6 model
        y_pred = y_pred_LLR2[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        specificity_sensitivity_sum = tpr + (1 - fpr)
        ind_max = np.argmax(specificity_sensitivity_sum)
        if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
            ind_max = 1
        opt_cutoff = thresholds[ind_max]
        AUC, _, AUC_05, AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[1],linestyle='-', label='LLR2 AUC: %.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
            ax1[i].legend(frameon=False, loc=(0.1, -0.02), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
        else:
            ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
            ax1[i].legend(frameon=False, loc=(0.2,-0.02), prop={'size': textSize},handlelength=1,handletextpad=0.1,
                      labelspacing = 0.2)
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