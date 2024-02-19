###############################################################################################
#Aim: Predictive power of NSCLC-specific LLR6 on other cancer types
#Description: ROC / AUC comparison between NSCLC-specific LLR6 vs. PDL1 vs. TMB on three other cancer types
#             (Fig. 7a).
#Run command: python 11.NSCLC_LLR6_on_otherCancers_ROC_AUC.py
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
from sklearn.metrics import roc_curve, auc

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")


def AUC_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auroc, threshold[ind_max]


if __name__ == "__main__":
    start_time = time.time()

    ########################## Read in data ##########################
    phenoNA = 'Response'
    LLRmodelNA = 'LLR6'
    featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age']
    xy_colNAs = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age'] + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    dataChowell = pd.concat([dataChowellTrain, dataChowellTest], axis=0)
    dataChowell_NSCLC = dataChowell.loc[dataChowell['CancerType'] == 'NSCLC',]
    dataChowell_Gastric = dataChowell.loc[dataChowell['CancerType'] == 'Gastric',]
    dataChowell_Mesothelioma = dataChowell.loc[dataChowell['CancerType'] == 'Mesothelioma',]
    dataChowell_Esophageal = dataChowell.loc[dataChowell['CancerType'] == 'Esophageal',]

    dataMorris_new = pd.read_excel(dataALL_fn, sheet_name='Morris_new', index_col=0)
    dataMorris_new2 = pd.read_excel(dataALL_fn, sheet_name='Morris_new2', index_col=0)
    dataMorris = pd.concat([dataMorris_new, dataMorris_new2], axis=0)
    dataMorris_Gastric = dataMorris.loc[dataMorris['CancerType'] == 'Gastric',]
    dataMorris_Mesothelioma = dataMorris.loc[dataMorris['CancerType'] == 'Mesothelioma',]
    dataMorris_Esophageal = dataMorris.loc[dataMorris['CancerType'] == 'Esophageal',]

    dataMSK_Gastric = pd.concat([dataChowell_Gastric[xy_colNAs], dataMorris_Gastric[xy_colNAs]], axis=0)
    dataMSK_Mesothelioma = pd.concat([dataChowell_Mesothelioma[xy_colNAs], dataMorris_Mesothelioma[xy_colNAs]], axis=0)
    dataMSK_Esophageal = pd.concat([dataChowell_Esophageal[xy_colNAs], dataMorris_Esophageal[xy_colNAs]], axis=0)

    dataLee = pd.read_excel(dataALL_fn, sheet_name='Lee_NSCLC', index_col=0)

    dataALL = [dataChowell_NSCLC, dataMSK_Gastric, dataMSK_Mesothelioma, dataMSK_Esophageal]
    for i in range(len(dataALL)):
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

    x_test_list = []
    y_test_list = []
    for c in dataALL:
        x_test_list.append(pd.DataFrame(c, columns=xy_colNAs))
        y_test_list.append(c[phenoNA])

    y_pred_LLR6 = []
    y_pred_PDL1 = []
    y_pred_TMB = []

    ###################### Read in LLR model params ######################
    fnIn = '../03.Results/16features/NSCLC/NSCLC_'+LLRmodelNA+'_10k_ParamCalculate.txt'
    params_data = open(fnIn,'r').readlines()
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
    scaler_sd.fit(x_test_list[0][featuresNA])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c[featuresNA])))
    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        y_pred_LLR6.append(y_pred_test)

    ########################## test PDL1 model performance ##########################
    modelNA = 'PDL1'
    for i in range(len(x_test_list)):
        y_pred_test = x_test_list[i]['PDL1_TPS(%)']
        y_pred_PDL1.append(y_pred_test)

    ########################## test TMB model performance ##########################
    modelNA = 'TMB' # TMB NLR Albumin Age
    for i in range(len(x_test_list)):
        y_pred_test = x_test_list[i][modelNA]
        y_pred_TMB.append(y_pred_test)



    ############################## save source data for figure ##############################
    dataset_list = []
    true_label_list = []
    LLR6_pred_list = []
    PDL1_pred_list = []
    TMB_pred_list = []
    dataset_unique = ["Gastric","Esophageal","Mesothelioma"]
    for i in range(3):
        LLR6_pred_list.extend(y_pred_LLR6[i+1])
        PDL1_pred_list.extend(y_pred_PDL1[i+1])
        TMB_pred_list.extend(y_pred_TMB[i+1])
        true_label_list.extend(y_test_list[i+1])
        dataset_list.extend([dataset_unique[i]]*len(y_test_list[i+1]))
    df = pd.DataFrame({"Cancer_type":dataset_list, "True_label":true_label_list, "NSCLC_LLR6_score":LLR6_pred_list, "PDL1":PDL1_pred_list, "TMB":TMB_pred_list})
    df.to_csv('../03.Results/source_data_fig08a.csv', index=False)



    ############################## Plot ##############################
    textSize = 8

    ############# Plot ROC curves ##############
    output_fig1 = '../03.Results/NSCLC_'+LLRmodelNA+'_PDL1_TMB_ROC_compare_on_otherCancerTypes.pdf'
    ax1 = [0] * 3
    fig1, ((ax1[0], ax1[1], ax1[2])) = plt.subplots(1, 3, figsize=(6.5, 1.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.2, hspace=0.35)

    for i in range(3):
        y_true = y_test_list[i+1]
        ###### LLR6 model
        y_pred = y_pred_LLR6[i+1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        specificity_sensitivity_sum = tpr + (1 - fpr)
        ind_max = np.argmax(specificity_sensitivity_sum)
        if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
            ind_max = 1
        opt_cutoff = thresholds[ind_max]
        AUC = auc(fpr, tpr)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='%s AUC: %.2f' % (LLRmodelNA[0:4],AUC))
        ###### PDL1 model
        y_pred = y_pred_PDL1[i+1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC = auc(fpr, tpr)
        ax1[i].plot(fpr, tpr, color= palette[2],linestyle='-', label='PDL1 AUC: %.2f' % (AUC))
        ###### TMB model
        y_pred = y_pred_TMB[i+1]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC = auc(fpr, tpr)
        ax1[i].plot(fpr, tpr, color= palette[3],linestyle='-', label='TMB AUC: %.2f' % (AUC))

        ax1[i].legend(frameon=False, loc=(0.4,-0.01), prop={'size': textSize},handlelength=1,handletextpad=0.1,
                      labelspacing = 0.2)
        ax1[i].set_xlim([-0.02, 1.02])
        ax1[i].set_ylim([-0.02, 1.02])
        ax1[i].set_yticks([0,0.5,1])
        ax1[i].set_xticks([0,0.5,1])
        if i > 0:
            ax1[i].set_yticklabels([])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1) # , dpi=300
    plt.close()
