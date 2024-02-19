###############################################################################################
#Aim: LLR6 vs. RF16(Chowell et al.) comparison
#Description: AUC comparison between LLR6 vs. RF16(Chowell et al.) on training and test sets
#             (Extended Data Fig. 2c,d).
#Run command, e.g.: python 05_7.PanCancer_LLR6_RF16NBT_AUC_compare.py train
###############################################################################################


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
import scipy.stats as stats
import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from scipy.stats import norm
from matplotlib.gridspec import GridSpec

palette = sns.color_palette("deep")

def delong_test(y_true, y_pred1, y_pred2, metric):
    n1 = len(y_pred1)
    n2 = len(y_pred2)
    if metric=='AUC':
        auc1 = roc_auc_score(y_true, y_pred1)
        auc2 = roc_auc_score(y_true, y_pred2)
    elif metric=='PRAUC':
        auc1 = average_precision_score(y_true, y_pred1)
        auc2 = average_precision_score(y_true, y_pred2)
    Q1 = auc1 * (1 - auc1)
    Q2 = auc2 * (1 - auc2)
    var = (Q1 / n1) + (Q2 / n2)
    Z_statistic = (auc1 - auc2) / np.sqrt(var)
    p_value = 2.0 * (1 - norm.cdf(abs(Z_statistic)))
    return p_value


if __name__ == "__main__":

    plot_train_or_test = sys.argv[1] #  train test

    ########################## Read in data ##########################
    phenoNA = 'Response'
    featuresNA_LLR6 = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16']
    featuresNA_RF16 = ["CancerType_grouped", "Albumin", "HED", "TMB", "FCNA", "BMI", "NLR", "Platelets", "HGB",
                       "Stage", "Age", "Drug","Chemo_before_IO", "HLA_LOH", "MSI", "Sex"]

    print('Raw data processing ...')
    dataALL_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
    dataChowellTrain = pd.read_excel(dataALL_fn, sheet_name='Chowell2015-2017', index_col=0)
    data_RF16_train = dataChowellTrain[featuresNA_RF16 + [phenoNA]]
    dataChowellTest = pd.read_excel(dataALL_fn, sheet_name='Chowell2018', index_col=0)
    if plot_train_or_test=='test':
        data_LLR6_test = dataChowellTest[featuresNA_LLR6+[phenoNA]]
        data_RF16_test = dataChowellTest[featuresNA_RF16 + [phenoNA]]
    elif plot_train_or_test=='train':
        data_LLR6_test = dataChowellTrain[featuresNA_LLR6 + [phenoNA]]
        data_RF16_test = dataChowellTrain[featuresNA_RF16 + [phenoNA]]

    data_LLR6_test = data_LLR6_test.astype(float)
    data_LLR6_test = data_LLR6_test.dropna(axis=0)
    TMB_upper = 50
    data_LLR6_test['TMB'] = [c if c < TMB_upper else TMB_upper for c in data_LLR6_test['TMB']]
    Age_upper = 85
    data_LLR6_test['Age'] = [c if c < Age_upper else Age_upper for c in data_LLR6_test['Age']]
    NLR_upper = 25
    data_LLR6_test['NLR'] = [c if c < NLR_upper else NLR_upper for c in data_LLR6_test['NLR']]
    x_test_LLR6 = data_LLR6_test[featuresNA_LLR6]
    y_test_LLR6 = data_LLR6_test[phenoNA]

    data_RF16_train = data_RF16_train.astype(float)
    data_RF16_train = data_RF16_train.dropna(axis=0)
    x_train_RF16 = data_RF16_train[featuresNA_RF16]
    y_train_RF16 = data_RF16_train[phenoNA]

    data_RF16_test = data_RF16_test.astype(float)
    data_RF16_test = data_RF16_test.dropna(axis=0)
    x_test_RF16 = data_RF16_test[featuresNA_RF16]
    y_test_RF16 = data_RF16_test[phenoNA]

    y_LLR6pred_test = []
    y_RF16pred_test = []
    ###################### test LLR6 model performance ######################
    fnIn = '../03.Results/6features/PanCancer/PanCancer_LLR6_10k_ParamCalculate.txt'
    params_data = open(fnIn,'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val
    scaler_sd = StandardScaler()
    scaler_sd.fit(x_test_LLR6[featuresNA_LLR6])
    scaler_sd.mean_ = np.array(params_dict['LLR_mean'])
    scaler_sd.scale_ = np.array(params_dict['LLR_scale'])
    x_test_scaled = pd.DataFrame(scaler_sd.transform(x_test_LLR6[featuresNA_LLR6]))
    clf = linear_model.LogisticRegression().fit(x_test_scaled, y_test_LLR6)
    clf.coef_ = np.array([params_dict['LLR_coef']])
    clf.intercept_ = np.array(params_dict['LLR_intercept'])
    y_LLR6pred_test = clf.predict_proba(x_test_scaled)[:, 1]

    ###################### test RF16 model performance ######################
    param_dict = {'n_estimators': 1000, 'min_samples_split': 2, 'min_samples_leaf': 20, 'max_depth': 8}
    clf = RandomForestClassifier(random_state=1, n_jobs=-1, **param_dict).fit(x_train_RF16,y_train_RF16)
    y_RF16pred_test = clf.predict_proba(x_test_RF16)[:, 1]

    ################ Test AUC and PRAUC difference p value between models ######
    p1 = delong_test(y_test_RF16, y_LLR6pred_test, y_RF16pred_test, 'AUC')
    print('LLR6 vs RF16 AUC p-val: %g' % (p1))
    p2 = delong_test(y_test_RF16, y_LLR6pred_test, y_RF16pred_test, 'PRAUC')
    print('LLR6 vs RF16 PRAUC p-val: %g' % (p2))

    ################ print source data ################
    print("True_label LLR6_score RF16_score")
    True_label = y_test_RF16.tolist()
    for i in range(len(True_label)):
        print(True_label[i],y_LLR6pred_test[i],y_RF16pred_test[i])

    ############################## Plot ##############################
    textSize = 8
    ############# Plot ROC curves ##############
    output_fig1 = '../03.Results/PanCancer_LLR6_vs_RF16NBT_AUC_AUPRC_compare_'+plot_train_or_test+'.pdf'
    ax1 = [0] * 2
    fig1, ((ax1[0], ax1[1])) = plt.subplots(1, 2, figsize=(3.7,1.7))
    fig1.subplots_adjust(left=0.15, bottom=0.25, right=0.97, top=0.96, wspace=0.37, hspace=0.5)


    i = 0
    y_true = y_test_RF16
    ###### LLR6 model
    fpr, tpr, thresholds = roc_curve(y_true, y_LLR6pred_test)
    AUC = roc_auc_score(y_true, y_LLR6pred_test)
    #ax1[i].plot([0, 1], [0, 1], 'k', alpha=0.5, linestyle='--')
    ax1[i].plot(fpr, tpr, color= palette[0],linestyle='-', label='LLR6 AUC: %.2f' % (AUC))
    ###### RF16 model
    fpr, tpr, thresholds = roc_curve(y_true, y_RF16pred_test)
    AUC = roc_auc_score(y_true, y_RF16pred_test)
    ax1[i].plot(fpr, tpr, color=palette[1], linestyle='-', label='RF16 AUC: %.2f' % (AUC))

    ax1[i].legend(frameon=False, loc=(0.05, 0), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                  labelspacing=0.2)
    #ax1[i].text(0.5, 0.4, 'p = %.2f' % p1)

    ax1[i].set_xlim([-0.02, 1.02])
    ax1[i].set_ylim([-0.02, 1.02])
    ax1[i].set_yticks([0,0.5,1])
    ax1[i].set_xticks([0,0.5,1])
    ax1[i].set_xlabel('1 - specificity')
    ax1[i].set_ylabel('Sensitivity')
    ax1[i].spines['right'].set_visible(False)
    ax1[i].spines['top'].set_visible(False)

    i = 1
    ###### LLR6 model
    precision, recall, thresholds = precision_recall_curve(y_true, y_LLR6pred_test)
    AUC = average_precision_score(y_true, y_LLR6pred_test)
    ax1[i].plot(recall, precision, color=palette[0], linestyle='-', label='LLR6 AUPRC: %.2f' % (AUC))
    ###### RF16 model
    precision, recall, thresholds = precision_recall_curve(y_true, y_RF16pred_test)
    AUC = average_precision_score(y_true, y_RF16pred_test)
    ax1[i].plot(recall, precision, color=palette[1], linestyle='-', label='RF16 AUPRC: %.2f' % (AUC))
    ax1[i].legend(frameon=False, loc=(0.05, 0), prop={'size': textSize}, handlelength=1, handletextpad=0.1,
                  labelspacing=0.2)

    ax1[i].set_xlim([-0.02, 1.02])
    ax1[i].set_ylim([-0.02, 1.02])
    ax1[i].set_yticks([0, 0.5, 1])
    ax1[i].set_xticks([0, 0.5, 1])
    ax1[i].set_xlabel('Recall')
    ax1[i].set_ylabel('Precision')
    ax1[i].spines['right'].set_visible(False)
    ax1[i].spines['top'].set_visible(False)

    fig1.savefig(output_fig1)
    plt.close()


    ##################### correlation scatter plot between predicted scores from LLR6 and RF16 #####################
    textSize = 8
    output_fig2 = '../03.Results/PanCancer_LLR6_vs_RF16NBT_scoreCorrelation_scatterPlot.pdf'
    # fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    # fig.subplots_adjust(left=0.27, bottom=0.25, right=0.93, top=0.96, wspace=0.35, hspace=0.5)

    fig = plt.figure(figsize=(3, 3))
    gs = GridSpec(4, 4, left=0.2, bottom=0.15, right=0.96, top=0.96, wspace=0, hspace=0)
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3])
    ax_marg_y = fig.add_subplot(gs[1:4, 3])

    x=y_LLR6pred_test
    y=y_RF16pred_test
    ax_joint.scatter(x, y, color='black', s=10)
    spearman_corr, _ = stats.spearmanr(x, y)
    textstr = f'r = {spearman_corr:.2f}' # \np = {p_value:.1e}
    ax_joint.text(0.15, 0.8, textstr, fontsize=textSize, color='black', backgroundcolor='white')

    binBoundaries = np.linspace(0, 100,21)/100
    ax_marg_x.hist(x, edgecolor='black', facecolor = '0.5',linewidth=1,bins=binBoundaries)
    ax_marg_y.hist(y, orientation="horizontal", edgecolor='black', facecolor = '0.5',linewidth=1,bins=binBoundaries)

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    # Set labels on joint
    ax_joint.set_xlabel('LLR6 score')
    ax_joint.set_ylabel('RF16 score')
    ax_joint.set_xlim([0, 1])
    ax_joint.set_ylim([0, 1])
    ax_joint.set_xticks([0, 0.5, 1])
    ax_joint.set_xticks([0, 0.5, 1])
    ax_marg_x.set_xlim([0, 1])
    ax_marg_y.set_ylim([0, 1])


    # Set labels on marginals
    # ax_marg_y.set_xlabel('Marginal x label')
    # ax_marg_x.set_ylabel('Marginal y label')
    ax_marg_y.spines['top'].set_visible(False)
    ax_marg_y.spines['right'].set_visible(False)
    ax_marg_y.spines['bottom'].set_visible(False)
    ax_marg_x.spines['left'].set_visible(False)
    ax_marg_x.spines['right'].set_visible(False)
    ax_marg_x.spines['top'].set_visible(False)
    ax_marg_x.set_yticks([])
    ax_marg_y.set_xticks([])


    # # Create your scatter plot
    # pp = sns.scatterplot(x=y_LLR6pred_test, y=y_RF16pred_test, color='black', s=10, ax=ax)
    # # Add density plot along the x-axis (top)
    # sns.kdeplot(y_LLR6pred_test, ax=pp.twinx(), color='red', legend=False)
    # # Add density plot along the y-axis (right)
    # sns.kdeplot(y_RF16pred_test, ax=pp.twiny(), color='blue', legend=False)
    # # Set labels for the density plots
    # pp.twinx().set_ylabel("Density", color="red")
    # pp.twiny().set_xlabel("Density", color="blue")


    # ax2.scatter(y_LLR6pred_test, y_RF16pred_test, color='black', s=10)
    # slope, intercept, r_value, p_value, std_err = stats.linregress(y_LLR6pred_test, y_RF16pred_test)
    # ax2.plot(y_LLR6pred_test, slope * y_LLR6pred_test + intercept, color='red')
    # spearman_corr, _ = stats.spearmanr(y_LLR6pred_test, y_RF16pred_test)
    # textstr = f'r = {spearman_corr:.2f}' # \np = {p_value:.1e}
    # ax2.text(0.15, 0.8, textstr, fontsize=textSize, color='black', backgroundcolor='white')
    # ax2.set_xlabel('LLR6 score')
    # ax2.set_ylabel('RF16 score')
    #
    # ax2.spines['right'].set_visible(False)
    # ax2.spines['top'].set_visible(False)
    # ax2.set_xticks([0, 0.5, 1])
    # ax2.set_yticks([0, 0.5, 1])

    fig.savefig(output_fig2) # , dpi=300
    plt.close()

