###############################################################################################
#Aim: Performance of pan-cancer LLR6 on MSK-ICB cohort vs MSK-nonICB cohort
#Description: Comparison of ROC / AUC of pan-cancer LR5noTMB on MSK-ICB cohort vs MSK-nonICB cohort.
#             (Extended Data Fig. 4e)
#Run command: python 07_4.PanCancer_LR5noTMB_LORIS_ROC_AUC_nonICB_vs_ICB_OS.py
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
from scipy.stats import norm
from sklearn.utils import resample

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")


def delong_test(y_true1, y_pred1, y_true2, y_pred2):
    not_nan_indices1 = ~np.isnan(y_true1.tolist())
    not_nan_indices2 = ~np.isnan(y_true2.tolist())
    y_true1 = y_true1[not_nan_indices1]
    y_pred1 = y_pred1[not_nan_indices1]
    y_true2 = y_true2[not_nan_indices2]
    y_pred2 = y_pred2[not_nan_indices2]
    n1 = len(y_pred1)
    n2 = len(y_pred2)
    auc1 = roc_auc_score(y_true1, y_pred1)
    auc2 = roc_auc_score(y_true2, y_pred2)
    Q1 = auc1 * (1 - auc1)
    Q2 = auc2 * (1 - auc2)
    var = (Q1 / n1) + (Q2 / n2)
    Z_statistic = (auc1 - auc2) / np.sqrt(var)
    p_value = 2.0 * (1 - norm.cdf(abs(Z_statistic)))
    return p_value


def AUC_with95CI_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    n_bootstrap = 10#00
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
    phenotype = ['OS0.5yr', 'OS1yr', 'OS2yr', 'OS3yr']
    cancer_type = 'all'
    modelNA = "LR5noTMB"
    featuresNA_1 = ['Systemic_therapy_history', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    xy_colNAs = ['Systemic_therapy_history', 'Albumin', 'NLR', 'Age', 'CancerType1',
                 'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                 'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                 'CancerType14', 'CancerType15', 'CancerType16'] + ['OS_Months', 'OS_Event', 'CancerType'] + phenotype

    print('Raw data processing ...')
    data_fn = '../02.Input/AllData.xlsx'
    dataICBtest1 = pd.read_excel(data_fn, sheet_name='Chowell_test', index_col=0)
    dataICBtest2 = pd.read_excel(data_fn, sheet_name='MSK1', index_col=0)
    dataICBtest = pd.concat([dataICBtest1, dataICBtest2], axis=0)
    temp_df = dataICBtest[xy_colNAs[0:21]].astype(float)
    rows_valid = (temp_df.isnull().sum(axis=1) == 0)
    dataICBtest = pd.concat([temp_df.loc[rows_valid, :], dataICBtest.loc[rows_valid, xy_colNAs[21:-4]]], axis=1)

    dataICBtest['OS0.5yr'] = np.nan
    dataICBtest.loc[(dataICBtest['OS_Months'] < 6) & (dataICBtest['OS_Event'] == 1),'OS0.5yr'] = 0
    dataICBtest.loc[dataICBtest['OS_Months'] >= 6, 'OS0.5yr'] = 1
    dataICBtest['OS1yr'] = np.nan
    dataICBtest.loc[(dataICBtest['OS_Months'] < 12) & (dataICBtest['OS_Event'] == 1), 'OS1yr'] = 0
    dataICBtest.loc[dataICBtest['OS_Months'] >= 12, 'OS1yr'] = 1
    dataICBtest['OS2yr'] = np.nan
    dataICBtest.loc[(dataICBtest['OS_Months'] < 24) & (dataICBtest['OS_Event'] == 1), 'OS2yr'] = 0
    dataICBtest.loc[dataICBtest['OS_Months'] >= 24, 'OS2yr'] = 1
    dataICBtest['OS3yr'] = np.nan
    dataICBtest.loc[(dataICBtest['OS_Months'] < 36) & (dataICBtest['OS_Event'] == 1), 'OS3yr'] = 0
    dataICBtest.loc[dataICBtest['OS_Months'] >= 36, 'OS3yr'] = 1

    dataNonICBtest = pd.read_excel(data_fn, sheet_name='MSK_nonICB', index_col=0)
    temp_df = dataNonICBtest[xy_colNAs[0:21]].astype(float)
    rows_valid = (temp_df.isnull().sum(axis=1) == 0)
    dataNonICBtest = pd.concat([temp_df.loc[rows_valid, :], dataNonICBtest.loc[rows_valid, xy_colNAs[21:-4]]], axis=1)

    dataNonICBtest['OS0.5yr'] = np.nan
    dataNonICBtest.loc[(dataNonICBtest['OS_Months'] < 6) & (dataNonICBtest['OS_Event'] == 1), 'OS0.5yr'] = 0
    dataNonICBtest.loc[dataNonICBtest['OS_Months'] >= 6, 'OS0.5yr'] = 1
    dataNonICBtest['OS1yr'] = np.nan
    dataNonICBtest.loc[(dataNonICBtest['OS_Months'] < 12) & (dataNonICBtest['OS_Event'] == 1), 'OS1yr'] = 0
    dataNonICBtest.loc[dataNonICBtest['OS_Months'] >= 12, 'OS1yr'] = 1
    dataNonICBtest['OS2yr'] = np.nan
    dataNonICBtest.loc[(dataNonICBtest['OS_Months'] < 24) & (dataNonICBtest['OS_Event'] == 1), 'OS2yr'] = 0
    dataNonICBtest.loc[dataNonICBtest['OS_Months'] >= 24, 'OS2yr'] = 1
    dataNonICBtest['OS3yr'] = np.nan
    dataNonICBtest.loc[(dataNonICBtest['OS_Months'] < 36) & (dataNonICBtest['OS_Event'] == 1), 'OS3yr'] = 0
    dataNonICBtest.loc[dataNonICBtest['OS_Months'] >= 36, 'OS3yr'] = 1

    dataALL = [dataICBtest, dataNonICBtest]

    if cancer_type == 'nonNSCLC':
        dataALL = [c.loc[c['CancerType11']==0,:] for c in dataALL]

    for i in range(len(dataALL)):
        temp_df = dataALL[i][xy_colNAs[0:21]].astype(float)
        rows_valid = (temp_df.isnull().sum(axis=1) == 0)
        dataALL[i] = pd.concat([temp_df.loc[rows_valid, :], dataALL[i].loc[rows_valid, xy_colNAs[21:]]], axis=1)

    print('ICB dataset size: ', dataICBtest.shape[0])
    print('Non-ICB dataset size: ', dataNonICBtest.shape[0])

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

    x_test_list = []
    y_test_list_list = [[],[],[],[]] # 0.5/1/2/3yrOS
    for c in dataALL:
        x_test_list.append(pd.DataFrame(c, columns=featuresNA_1))
        y_test_list_list[0].append(c[phenotype[0]])
        y_test_list_list[1].append(c[phenotype[1]])
        y_test_list_list[2].append(c[phenotype[2]])
        y_test_list_list[3].append(c[phenotype[3]])

    y_LLR6pred_test_list = []

    ###################### Read in LLR6-ICB (1yrOS) model params ######################
    fnIn = '../03.Results/6features/PanCancer/PanCancer_'+modelNA+'_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val
    ########################## test LLR6 model performance ##########################
    x_test_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_list[0])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c)))
    # Remove rows with NaN in y
    not_nan_indices = ~np.isnan(y_test_list_list[0][0].tolist())
    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0].loc[not_nan_indices,], y_test_list_list[0][0][not_nan_indices])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        y_LLR6pred_test_list.append(y_pred_test)
        dataALL[i]['LLR6'] = y_pred_test
        dataALL[i].to_csv('../03.Results/'+modelNA+'_predict_nonICB_vs_ICB_'+ cancer_type + '_Dataset'+str(i+1)+'.csv', index=True)

    ################ Test AUC difference p value between models ######
    pval_list = []
    for i in range(len(phenotype)):
        pval = delong_test(y_test_list_list[i][0], y_LLR6pred_test_list[0], y_test_list_list[i][1], y_LLR6pred_test_list[1])
        print('%s: LLR6 ICB vs nonICB p-val: %g. '%(phenotype[i],pval))
        pval_list.append(pval)

    ############################## Plot ROC ##############################
    textSize = 8
    ############# Plot ROC curves ##############
    output_fig1 = '../03.Results/PanCancer_'+modelNA+'_LORIS_ROC_AUC_nonICB_vs_ICB_OS.pdf'
    rows, cols = 1, len(phenotype)
    fig1, ax1 = plt.subplots(rows, cols, figsize=(len(phenotype)*2+0.5, 2))
    ax1 = ax1.flatten()

    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.92, top=0.96, wspace=0.3, hspace=0.5)

    for i in range(len(phenotype)):
        y_true1 = y_test_list_list[i][0]
        y_pred1 = y_LLR6pred_test_list[0]
        y_true2 = y_test_list_list[i][1]
        y_pred2 = y_LLR6pred_test_list[1]

        not_nan_indices1 = ~np.isnan(y_true1.tolist())
        not_nan_indices2 = ~np.isnan(y_true2.tolist())
        y_true1 = y_true1[not_nan_indices1]
        y_pred1 = y_pred1[not_nan_indices1]
        y_true2 = y_true2[not_nan_indices2]
        y_pred2 = y_pred2[not_nan_indices2]

        # ICB
        fpr, tpr, thresholds = roc_curve(y_true1, y_pred1)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true1, y_pred1)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='ICB AUC: %.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        else:
            ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        # Non-ICB
        fpr, tpr, thresholds = roc_curve(y_true2, y_pred2)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true2, y_pred2)
        if not i:
            ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-',
                        label='Non-ICB AUC: %.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
            ax1[i].legend(frameon=False, loc=(0.1, 0.03), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
        else:
            ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC, AUC_05, AUC_95))
            ax1[i].legend(frameon=False, loc=(0.35,0.03), prop={'size': textSize},handlelength=1,handletextpad=0.1,
                      labelspacing = 0.2)
        ax1[i].set_xlim([-0.02, 1.02])
        ax1[i].set_ylim([-0.02, 1.02])
        ax1[i].set_yticks([0,0.5,1])
        ax1[i].set_xticks([0,0.5,1])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1)
    plt.close()
