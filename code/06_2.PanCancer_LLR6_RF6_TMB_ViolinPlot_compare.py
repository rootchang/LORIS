###############################################################################################
#Aim: Pan-cancer LLR6 vs. RF6 vs. TMB comparison
#Description: Comparison LLR6 vs. RF6 vs. TMB scores between responders and non-responders on training and multiple
#             test sets. Specifically
#             1) Models on all patients
#             2) Models on non-NSCLC patients
#             (Fig. 2b; Extended Data Fig. 6b).
#Run command, e.g.: python 06_2.PanCancer_LLR6_RF6_TMB_ViolinPlot_compare.py all
###############################################################################################


import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import sys


if __name__ == "__main__":
    start_time = time.time()

    plot_df = pd.DataFrame()
    plot_LLR6_df = pd.DataFrame()
    plot_data_LLR6 = []
    LLRmodelNA = 'LLR6'
    cancer_type = sys.argv[1] # 'all'    'nonNSCLC'
    print('Raw data read in ...')
    fnIn = '../03.Results/PanCancer_' + cancer_type + '_' + LLRmodelNA + '_Scaler(StandardScaler)_prediction.xlsx'
    start_set = 0
    if start_set:
        output_fig_fn = '../03.Results/PanCancer_'+LLRmodelNA+'_RF6_TMB_N_NR_score_violin_testOnly.pdf'
    else:
        output_fig_fn = '../03.Results/PanCancer_'+LLRmodelNA+'_RF6_TMB_N_NR_score_violin_all.pdf'
    for sheet_i in range(start_set,6):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_LLR6=np.array(data['y_pred'].tolist())
        y_true=np.array(data['y'].tolist())
        temp_df = pd.DataFrame()
        temp_df['Dataset'] = ['Test set '+str(sheet_i)] * sum(y_true)
        temp_df['Response'] = ['R'] * sum(y_true)
        temp_df['LLR6_score'] = y_pred_LLR6[y_true == 1]
        plot_data_LLR6.append(y_pred_LLR6[y_true == 1])

        plot_LLR6_df = pd.concat([plot_LLR6_df, temp_df], axis=0)
        temp_df = pd.DataFrame()
        temp_df['Dataset'] = ['Test set '+str(sheet_i)] * (len(y_true) - sum(y_true))
        temp_df['Response'] = ['NR'] * (len(y_true) - sum(y_true))
        temp_df['LLR6_score'] = y_pred_LLR6[y_true == 0]
        plot_data_LLR6.append(y_pred_LLR6[y_true == 0])
        plot_LLR6_df = pd.concat([plot_LLR6_df, temp_df], axis=0)
    plot_LLR6_df = plot_LLR6_df.reset_index(drop=True)

    plot_RF6_df = pd.DataFrame()
    plot_data_RF6 = []
    fnIn = '../03.Results/PanCancer_' + cancer_type + '_' + 'RF6_Scaler(None)_prediction.xlsx'
    for sheet_i in range(start_set,6):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_RF6 = np.array(data['y_pred'].tolist())
        y_true = np.array(data['y'].tolist())
        temp_df = pd.DataFrame()
        temp_df['RF6_score'] = y_pred_RF6[y_true == 1]
        plot_data_RF6.append(y_pred_RF6[y_true == 1])
        plot_RF6_df = pd.concat([plot_RF6_df, temp_df], axis=0)
        temp_df = pd.DataFrame()
        temp_df['RF6_score'] = y_pred_RF6[y_true == 0]
        plot_data_RF6.append(y_pred_RF6[y_true == 0])
        plot_RF6_df = pd.concat([plot_RF6_df, temp_df], axis=0)
    plot_RF6_df = plot_RF6_df.reset_index(drop=True)

    plot_TMB_df = pd.DataFrame()
    plot_data_TMB = []
    fnIn = '../03.Results/PanCancer_' +  cancer_type + '_TMB_Scaler(None)_prediction.xlsx'
    for sheet_i in range(start_set,6):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_TMB = np.array(data['y_pred'].tolist())
        y_true = np.array(data['y'].tolist())
        temp_df = pd.DataFrame()
        temp_df['TMB'] = np.log2(y_pred_TMB[y_true == 1]+1)
        plot_data_TMB.append(y_pred_TMB[y_true == 1])
        plot_TMB_df = pd.concat([plot_TMB_df, temp_df], axis=0)
        temp_df = pd.DataFrame()
        temp_df['TMB'] = np.log2(y_pred_TMB[y_true == 0]+1)
        plot_data_TMB.append(y_pred_TMB[y_true == 0])
        plot_TMB_df = pd.concat([plot_TMB_df, temp_df], axis=0)
    plot_TMB_df = plot_TMB_df.reset_index(drop=True)

    plot_df = pd.concat([plot_LLR6_df, plot_RF6_df], axis=1)
    plot_df = pd.concat([plot_df, plot_TMB_df], axis=1)

    for i in range(6):
        x = plot_data_LLR6[i*2]
        y = plot_data_LLR6[i*2+1]
        statistic, p_value = mannwhitneyu(x, y, alternative='greater')
        print(LLRmodelNA, ' p-value: ', p_value)
    for i in range(6):
        x = plot_data_RF6[i*2]
        y = plot_data_RF6[i*2+1]
        statistic, p_value = mannwhitneyu(x, y, alternative='greater')
        print('RF6 p-value: ', p_value)
    for i in range(6):
        x = plot_data_TMB[i*2]
        y = plot_data_TMB[i*2+1]
        statistic, p_value = mannwhitneyu(x, y, alternative='greater')
        print('TMB p-value: ', p_value)

    ################ violin & box plot for scores distribution among R and NR
    fontSize = 10
    plt.rcParams['font.size'] = fontSize
    plt.rcParams["font.family"] = "Arial"
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.7))
    plt.subplots_adjust(left=0.08, bottom=0.34, right=0.97, top=0.99, wspace=0.4, hspace=0.45)

    boxplot_linewidth = 0.5
    my_palette = {"R": "g", "NR": "0.5"}
    xlabel = ['Chowell train','Chowell test','MSK1','MSK2','Kato et al.','Pradat et al.']
    ###################################### axes[0]: LLR6 ######################################
    graph = sns.violinplot(y="LLR6_score", x='Dataset', hue="Response", data=plot_df,
                           palette=my_palette, linewidth=0.5, saturation=0.75,
                           scale="width", ax=axes[0], zorder=2)
    graph.legend_.remove()
    axes[0].set_xticklabels(xlabel, rotation=30, ha='right')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('LORIS')

    ###################################### axes[1]: RF6 ######################################
    graph = sns.violinplot(y="RF6_score", x='Dataset', hue="Response", data=plot_df,
                           palette=my_palette, linewidth=0.5, saturation=0.75,
                           scale="width", ax=axes[1], zorder=2)
    graph.legend_.remove()
    axes[1].set_xticklabels(xlabel, rotation=30, ha='right')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('RF6 score')

    ###################################### axes[2]: TMB ######################################
    graph = sns.violinplot(y="TMB", x='Dataset', hue="Response", data=plot_df,
                           palette=my_palette, linewidth=0.5, saturation=0.75,
                           scale="width", ax=axes[2], zorder=2)
    graph.legend_.remove()
    axes[2].set_xticklabels(xlabel, rotation=30, ha='right')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('log2(TMB+1)')

    for i in range(3):
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['top'].set_visible(False)
    axes[0].set_yticks([0,0.5,1])
    axes[1].set_yticks([0, 0.5, 1])
    axes[2].set_yticks([0, 3, 6])
    axes[0].set_ylim([-0.1, 1.1])
    axes[1].set_ylim([-0.1, 1.1])
    axes[2].set_ylim([-1, 7])
    fig.savefig(output_fig_fn)
    plt.close()