###############################################################################################
#Aim: LLR6 vs. RF6 vs. TMB comparison
#Description: Multiple metric comparison between LLR6 vs. RF6 vs. TMB on training and multiple test sets. Specifically
#             1) Models on all patients
#             2) Models on non-NSCLC patients
#             (Fig. 2a,c; Extended Data Fig. 7a).
#Run command, e.g.: python 06_1.PanCancer_LLR6_RF6_TMB_multiMetric_compare.py all
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix, average_precision_score, roc_auc_score
from scipy.stats import norm
from sklearn.utils import resample

plt.rcParams.update({'font.size': 10})
plt.rcParams["font.family"] = "Arial"
palette = ["#377EB8", "#FF7F00", "#4D9943", "#E7BA52", "#999999", "#F7CAC9"]
sns.set_palette(palette)
#palette = sns.color_palette("deep")


def AUC_calculator(y, y_pred):
    fpr, tpr, threshold = roc_curve(y, y_pred, pos_label=1)
    auroc = auc(fpr, tpr)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    return auroc, threshold[ind_max]


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


def AUPRC_with95CI_calculator(y, y_pred):
    prec, recall, threshold = precision_recall_curve(y, y_pred)
    AUPRC = auc(recall, prec)
    specificity_sensitivity_sum = recall + (1 - prec)
    ind_max = np.argmax(specificity_sensitivity_sum)
    n_bootstrap = 1000
    auc_values = []
    for _ in range(n_bootstrap):
        y_true_bootstrap, y_scores_bootstrap = resample(y, y_pred, replace=True)
        AUPRC_temp = average_precision_score(y_true_bootstrap, y_scores_bootstrap)
        auc_values.append(AUPRC_temp)
    lower_auprc = np.percentile(auc_values, 2.5)
    upper_auprc = np.percentile(auc_values, 97.5)
    return AUPRC, threshold[ind_max], lower_auprc, upper_auprc


if __name__ == "__main__":
    start_time = time.time()

    CPU_num = -1
    randomSeed = 1
    fix_cutoff = 1
    cutoff_value_RF6 = 0.27
    cutoff_value_TMB = 10

    phenoNA = 'Response'
    LLRmodelNA = 'LLR6'
    cancer_type = sys.argv[1]  # 'all'   'nonNSCLC'
    model_type = 'all_'

    ########################## Read in data ##########################
    cutoff_value_LLRx = 0.5
    featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    featuresNA_RF6 = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age', 'CancerType_grouped']
    xy_colNAs = ['TMB', 'Systemic_therapy_history', 'Albumin', 'NLR', 'Age', 'CancerType1',
                 'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                 'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                 'CancerType14', 'CancerType15', 'CancerType16'] + ['CancerType_grouped'] + [phenoNA]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/AllData.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell_test', index_col=0)
    dataMSK1 = pd.read_excel(dataALL_fn, sheet_name='MSK1', index_col=0)  # MSK1
    dataMSK12 = pd.read_excel(dataALL_fn, sheet_name='MSK12', index_col=0)  # MSK12
    dataKato = pd.read_excel(dataALL_fn, sheet_name='Kato_panCancer', index_col=0)
    dataPradat = pd.read_excel(dataALL_fn, sheet_name='Pradat_panCancer', index_col=0)

    dataALL = [dataChowellTrain, dataChowellTest, dataMSK1, dataMSK12, dataKato, dataPradat]

    if cancer_type == 'nonNSCLC':
        dataALL = [c.loc[c['CancerType11']==0,:] for c in dataALL]

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
    x_test_RF6_list = []
    y_test_list = []
    for c in dataALL:
        x_test_list.append(pd.DataFrame(c, columns=featuresNA))
        x_test_RF6_list.append(pd.DataFrame(c, columns=featuresNA_RF6))
        y_test_list.append(c[phenoNA])


    y_pred_LLRx = []
    AUPRC_LLRx = []
    Sensitivity_LLRx = []
    Specificity_LLRx = []
    Accuracy_LLRx = []
    PPV_LLRx = []
    NPV_LLRx = []
    F1_LLRx = []
    OddsRatio_LLRx = []

    y_pred_RF6 = []
    AUPRC_RF6 = []
    Sensitivity_RF6 = []
    Specificity_RF6 = []
    Accuracy_RF6 = []
    PPV_RF6 = []
    NPV_RF6 = []
    F1_RF6 = []
    OddsRatio_RF6 = []


    y_pred_TMB = []
    AUPRC_TMB = []
    Sensitivity_TMB = []
    Specificity_TMB = []
    Accuracy_TMB = []
    PPV_TMB = []
    NPV_TMB = []
    F1_TMB = []
    OddsRatio_TMB = []

    ###################### Read in LLRx model params ######################
    fnIn = '../03.Results/6features/PanCancer/PanCancer_'+model_type+LLRmodelNA+'_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val

    y_LLR6pred_test_list = []
    y_RF6pred_test_list = []
    y_TMBpred_test_list = []

    ########################## test LLRx model performance ##########################
    x_test_scaled_list = []
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_list[0])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    for c in x_test_list:
        x_test_scaled_list.append(pd.DataFrame(scaler_sd.transform(c)))

    clf = linear_model.LogisticRegression().fit(x_test_scaled_list[0], y_test_list[0])
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])

    print('LLRx_meanParams10000:')
    fnOut = '../03.Results/PanCancer_'+ cancer_type + '_' + LLRmodelNA + '_Scaler(' + 'StandardScaler' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_scaled_list)):
        y_pred_test = clf.predict_proba(x_test_scaled_list[i])[:, 1]
        y_LLR6pred_test_list.append(y_pred_test)
        dataALL[i][LLRmodelNA] = y_pred_test
        dataALL[i].to_csv('../03.Results/PanCancermodel_'+LLRmodelNA+'_Dataset'+str(i+1)+'.csv', index=True)
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response',LLRmodelNA]]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if fix_cutoff:
            score = cutoff_value_LLRx
        else:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        AUPRC = average_precision_score(y_test_list[i], y_pred_test)
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)
        Specificity = tn / (tn + fp)
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        F1 = 2*PPV*Sensitivity/(PPV+Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_LLRx.append(y_pred_test)
        AUPRC_LLRx.append(AUPRC)
        Sensitivity_LLRx.append(Sensitivity)
        Specificity_LLRx.append(Specificity)
        Accuracy_LLRx.append(Accuracy)
        PPV_LLRx.append(PPV)
        NPV_LLRx.append(NPV)
        F1_LLRx.append(F1)
        OddsRatio_LLRx.append(OddsRatio)

    ########################## test RF6 model performance ##########################
    modelNA = 'RF6'
    params = {'n_estimators': 900, 'min_samples_split': 20, 'min_samples_leaf': 8, 'max_depth': 8}
    clf = RandomForestClassifier(random_state=randomSeed, n_jobs=CPU_num, **params).fit(x_test_RF6_list[0], y_test_list[0])

    print('RF6:')
    fnOut = '../03.Results/PanCancer_' + cancer_type + '_' + modelNA + '_Scaler(' + 'None' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_RF6_list)):
        y_pred_test = clf.predict_proba(x_test_RF6_list[i])[:, 1]
        y_RF6pred_test_list.append(y_pred_test)
        dataALL[i][modelNA] = y_pred_test
        dataALL[i].to_csv('../03.Results/PanCancermodel_'+modelNA+'_Dataset'+str(i+1)+'.csv', index=True)
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response',modelNA]]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if fix_cutoff:
            score = cutoff_value_RF6
        else:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        AUPRC = average_precision_score(y_test_list[i], y_pred_test)
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)
        Specificity = tn / (tn + fp)
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        F1 = 2 * PPV * Sensitivity / (PPV + Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_RF6.append(y_pred_test)
        AUPRC_RF6.append(AUPRC)
        Sensitivity_RF6.append(Sensitivity)
        Specificity_RF6.append(Specificity)
        Accuracy_RF6.append(Accuracy)
        PPV_RF6.append(PPV)
        NPV_RF6.append(NPV)
        F1_RF6.append(F1)
        OddsRatio_RF6.append(OddsRatio)

    ########################## test TMB model performance ##########################
    modelNA = 'TMB'

    print(modelNA+':')
    fnOut = '../03.Results/PanCancer_' + cancer_type + '_' + modelNA + '_Scaler(' + 'None' + ')_prediction.xlsx'
    dataALL[0].to_excel(fnOut, sheet_name='0')
    for i in range(len(x_test_RF6_list)):
        y_pred_test = x_test_RF6_list[i][modelNA]
        y_TMBpred_test_list.append(y_pred_test)
        dataALL[i][modelNA] = y_pred_test
        dataALL[i].to_csv('../03.Results/PanCancermodel_'+modelNA+'_Dataset'+str(i+1)+'.csv', index=True)
        AUC_test, score_test = AUC_calculator(y_test_list[i], y_pred_test)
        print('   Dataset %d: %5.3f (n=%d) %8.3f' % (i+1, AUC_test, len(y_pred_test), score_test))

        content = dataALL[i].loc[:,['Response',modelNA]]
        content.rename(columns=dict(zip(content.columns, ['y','y_pred'])), inplace=True)
        with pd.ExcelWriter(fnOut, engine="openpyxl", mode='a',if_sheet_exists="replace") as writer:
            content.to_excel(writer, sheet_name=str(i))

        if not fix_cutoff:
            AUC, score = AUC_calculator(y_test_list[i], y_pred_test)
        else:
            score = cutoff_value_TMB
        y_pred_01 = [int(c >= score) for c in y_pred_test]
        AUPRC = average_precision_score(y_test_list[i], y_pred_test)
        tn, fp, fn, tp = confusion_matrix(y_test_list[i], y_pred_01).ravel()
        Sensitivity = tp / (tp + fn)
        Specificity = tn / (tn + fp)
        Accuracy = (tp + tn) / (tp + tn + fp + fn)
        PPV = tp / (tp + fp)
        NPV = tn / (tn + fn)
        F1 = 2 * PPV * Sensitivity / (PPV + Sensitivity)
        OddsRatio = (tp / (tp + fp)) / (fn / (tn + fn))

        y_pred_TMB.append(y_pred_test)
        AUPRC_TMB.append(AUPRC)
        Sensitivity_TMB.append(Sensitivity)
        Specificity_TMB.append(Specificity)
        Accuracy_TMB.append(Accuracy)
        PPV_TMB.append(PPV)
        NPV_TMB.append(NPV)
        F1_TMB.append(F1)
        OddsRatio_TMB.append(OddsRatio)


    ############################## Plot ROC curves ##############################
    textSize = 8
    output_fig1 = '../03.Results/PanCancer_'+LLRmodelNA+'_RF6_TMB_ROC_'+cancer_type+'.pdf'
    ax1 = [0] * 6
    fig1, ((ax1[0], ax1[1], ax1[2]), (ax1[3], ax1[4], ax1[5])) = plt.subplots(2, 3, figsize=(6.5, 3.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.5)

    for i in range(6):
        y_true = y_test_list[i]
        ###### LLRx model
        y_pred = y_pred_LLRx[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='%s AUC: %.2f (%.2f, %.2f)' % (LLRmodelNA[0:4], AUC,AUC_05,AUC_95))
        else:
            ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        ###### RF6 model
        y_pred = y_pred_RF6[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[2],linestyle='-', label='RF6 AUC: %.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        else:
            ax1[i].plot(fpr, tpr, color= palette[2],linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
        ###### TMB model
        y_pred = y_pred_TMB[i]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        AUC,_,AUC_05,AUC_95 = AUC_with95CI_calculator(y_true, y_pred)
        if not i:
            ax1[i].plot(fpr, tpr, color= palette[3],linestyle='-', label='TMB AUC: %.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
            ax1[i].legend(frameon=False, loc=(0.2, -0.02), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
        else:
            ax1[i].plot(fpr, tpr, color= palette[3],linestyle='-', label='%.2f (%.2f, %.2f)' % (AUC,AUC_05,AUC_95))
            ax1[i].legend(frameon=False, loc=(0.4, -0.02), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                          labelspacing=0.2)
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


    ############# Plot metrics barplot ##############
    print('LLR6_AUPRC', ' '.join([str(c) for c in AUPRC_LLRx]))
    print('RF6_AUPRC', ' '.join([str(c) for c in AUPRC_RF6]))
    print('TMB_AUPRC', ' '.join([str(c) for c in AUPRC_TMB]))

    print('LLR6_odds_ratio', ' '.join([str(c) for c in OddsRatio_LLRx]))
    print('RF6_odds_ratio', ' '.join([str(c) for c in OddsRatio_RF6]))
    print('TMB_odds_ratio', ' '.join([str(c) for c in OddsRatio_TMB]))

    print('LLR6_Accuracy', ' '.join([str(c) for c in Accuracy_LLRx]))
    print('RF6_Accuracy', ' '.join([str(c) for c in Accuracy_RF6]))
    print('TMB_Accuracy', ' '.join([str(c) for c in Accuracy_TMB]))

    print('LLR6_F1', ' '.join([str(c) for c in F1_LLRx]))
    print('RF6_F1', ' '.join([str(c) for c in F1_RF6]))
    print('TMB_F1', ' '.join([str(c) for c in F1_TMB]))

    print('LLR6_PPV', ' '.join([str(c) for c in PPV_LLRx]))
    print('RF6_PPV', ' '.join([str(c) for c in PPV_RF6]))
    print('TMB_PPV', ' '.join([str(c) for c in PPV_TMB]))

    print('LLR6_NPV', ' '.join([str(c) for c in NPV_LLRx]))
    print('RF6_NPV', ' '.join([str(c) for c in NPV_RF6]))
    print('TMB_NPV', ' '.join([str(c) for c in NPV_TMB]))

    print('LLR6_Specificity', ' '.join([str(c) for c in Specificity_LLRx]))
    print('RF6_Specificity', ' '.join([str(c) for c in Specificity_RF6]))
    print('TMB_Specificity', ' '.join([str(c) for c in Specificity_TMB]))

    print('LLR6_Sensitivity', ' '.join([str(c) for c in Sensitivity_LLRx]))
    print('RF6_Sensitivity', ' '.join([str(c) for c in Sensitivity_RF6]))
    print('TMB_Sensitivity', ' '.join([str(c) for c in Sensitivity_TMB]))


    output_fig_fn = '../03.Results/PanCancer_'+LLRmodelNA+'_RF6_TMB_MultiMetric_'+cancer_type+'.pdf'
    ax1 = [0] * 8
    fig1, ((ax1[0], ax1[1]), (ax1[2], ax1[3]), (ax1[4], ax1[5]), (ax1[6], ax1[7])) = plt.subplots(4, 2, figsize=(6.5, 6.5))
    fig1.subplots_adjust(left=0.08, bottom=0.15, right=0.97, top=0.96, wspace=0.3, hspace=0.55)
    barWidth = 0.2
    color_list = [palette[0], palette[2], palette[3]]
    modelNA_list = [LLRmodelNA, 'RF6', 'TMB']
    metricsNA_list = ['AUPRC', 'Odds ratio', 'Accuracy', 'F1 score','PPV','NPV', 'Specificity', 'Sensitivity']
    LLRx_data = [AUPRC_LLRx, OddsRatio_LLRx, Accuracy_LLRx, F1_LLRx, PPV_LLRx, NPV_LLRx, Specificity_LLRx, Sensitivity_LLRx]
    RF6_data = [AUPRC_RF6, OddsRatio_RF6, Accuracy_RF6, F1_RF6, PPV_RF6, NPV_RF6, Specificity_RF6, Sensitivity_RF6]
    TMB_data = [AUPRC_TMB, OddsRatio_TMB, Accuracy_TMB, F1_TMB, PPV_TMB, NPV_TMB, Specificity_TMB, Sensitivity_TMB]
    for i in range(8):
        bh11 = ax1[i].bar(np.array([0, 1, 2, 3, 4, 5]) + barWidth * 1, LLRx_data[i],
                       color=color_list[0], width=barWidth, edgecolor='k', label=modelNA_list[0])
        bh12 = ax1[i].bar(np.array([0, 1, 2, 3, 4, 5]) + barWidth * 2, RF6_data[i],
                       color=color_list[1], width=barWidth, edgecolor='k', label=modelNA_list[1])
        bh13 = ax1[i].bar(np.array([0, 1, 2, 3, 4, 5]) + barWidth * 3, TMB_data[i],
                       color=color_list[2], width=barWidth, edgecolor='k', label=modelNA_list[2])

        ax1[i].set_xticks(np.array([0, 1, 2, 3, 4, 5]) + barWidth * 2)
        ax1[i].set_xticklabels([])
        if i in [0,2]:
            ax1[i].legend(frameon=False, loc=(0.05, 0.8), prop={'size': textSize}, handlelength=1, ncol=3,
                      handletextpad=0.1, labelspacing=0.2)
        ax1[i].set_xlim([0, 6])
        if i == 1:
            ax1[i].set_ylim([0, 7])
            ax1[i].set_yticks([0,2,4,6])
            ax1[i].axhline(y=1, color='grey', linestyle='--')
        else:
            ax1[i].set_ylim([0, 1])
            ax1[i].set_yticks([0, 0.5, 1])
        ax1[i].spines['right'].set_visible(False)
        ax1[i].spines['top'].set_visible(False)
        ax1[i].set_ylabel(metricsNA_list[i])
        ax1[i].tick_params('x', length=0, width=0, which='major')
    fig1.savefig(output_fig_fn)
    plt.close()