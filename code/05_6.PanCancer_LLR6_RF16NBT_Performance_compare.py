###############################################################################################
#Aim: LLR6 vs. RF16(Chowell et al.) comparison
#Description: Performance and delta performance comparison between LLR6 vs. RF16(Chowell et al.) in cross-validation
#             (Extended Data Fig. 2a,b).
#Run command, e.g.: python 05_6.PanCancer_LLR6_RF16NBT_Performance_compare.py test
###############################################################################################


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import sys
import scipy.stats as stats


def sig_str_calculator(y1, y2, pval, direction):
    if direction == 'greater':
        if np.mean(y1) <= np.mean(y2):
            return ''
    if direction == 'less':
        if np.mean(y1) >= np.mean(y2):
            return ''
    if pval < 0.05:
        if pval < 1e-300:
            return 'p < 1.0e-300'
        else:
            return 'p = {:.1e}'.format(pval)
    else:
        return ''


if __name__ == "__main__":
    plot_train_test_delta = sys.argv[1] # 'train', 'test', 'delta'

    if plot_train_test_delta in ['train', 'test']:
        sig_direction = 'greater'
    else:
        sig_direction = 'less'

    fontSize  = 10
    plt.rcParams['font.size'] = fontSize
    plt.rcParams["font.family"] = "Arial"

    fig, axes = plt.subplots(1, 5, figsize=(6.5, 1.5))
    plt.subplots_adjust(left=0.1, bottom=0.28, right=0.98, top=0.99, wspace=0.3, hspace=0.45)

    palette = sns.color_palette("deep")
    sns.set_style('white')

    print('Raw data reading in ...')
    resampleNUM = 10000

    try:
        randomSeed = int(sys.argv[1])
    except:
        randomSeed = 1
    CPU_num = -1
    N_repeat_KFold_paramTune = 1
    N_repeat_KFold = 2000
    info_shown = 1
    Kfold = 5
    resampleNUM = N_repeat_KFold*Kfold

    performance_df = pd.DataFrame()
    MLM_performance_dict = {}

    MLM_list = ['LLR6', 'RF16_NBT']
    SCALE_list = ['StandardScaler', 'None']

    for i in range(2):
        MLM = MLM_list[i]
        SCALE = SCALE_list[i]
        temp_df = pd.DataFrame()
        temp_df['method'] = [MLM] * resampleNUM
        if MLM == 'RF16_NBT':
            fnIn = '../03.Results/16features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + \
                   str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
        else:
            fnIn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + \
                str(Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
        dataIn = open(fnIn, 'r').readlines()
        header = dataIn[0].strip().split('\t')
        position = header.index('params')
        data = dataIn[1].strip().split('\t')
        data = data[(position + 2):]
        data = [float(c) for c in data]
        temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
        temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
        temp_df['AUPRC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
        temp_df['AUPRC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
        temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
        temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
        temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
        temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
        temp_df['MCC_test'] = data[(resampleNUM * 8 + 3 * 4 + 2 * 4):(resampleNUM * 9 + 3 * 4 + 2 * 4)]
        temp_df['MCC_train'] = data[(resampleNUM * 9 + 3 * 5 + 2 * 4):(resampleNUM * 10 + 3 * 5 + 2 * 4)]
        temp_df['BA_test'] = data[(resampleNUM * 10 + 3 * 5 + 2 * 5):(resampleNUM * 11 + 3 * 5 + 2 * 5)]
        temp_df['BA_train'] = data[(resampleNUM * 11 + 3 * 6 + 2 * 5):(resampleNUM * 12 + 3 * 6 + 2 * 5)]
        temp_df['Performance_test'] = (temp_df['AUC_test'] * temp_df['AUPRC_test'] * temp_df['Accuracy_test'] * temp_df[
            'F1_test']) ** (1 / 4)
        temp_df['Performance_train'] = (temp_df['AUC_train'] * temp_df['AUPRC_train'] * temp_df['Accuracy_train'] * temp_df[
            'F1_train']) ** (1 / 4)
        temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
        temp_df['AUPRC_delta'] = temp_df['AUPRC_train'] - temp_df['AUPRC_test']
        temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
        temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
        temp_df['MCC_delta'] = temp_df['MCC_train'] - temp_df['MCC_test']
        temp_df['BA_delta'] = temp_df['BA_train'] - temp_df['BA_test']
        temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
        performance_df = pd.concat([performance_df, temp_df], axis=0)

    ###################################### plot ######################################
    fig1, ax = plt.subplots(1, 1, figsize=(6.5, 2.5))
    fig1.subplots_adjust(left=0.08, bottom=0.35, right=0.97, top=0.96, wspace=0.3, hspace=0.55)
    barWidth = 0.3
    color_list = [palette[0], palette[2]]
    x_str = 'AUC_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ymax = max(max(y1),max(y2))
    ax.bar(0.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k', label='LLR6')
    ax.bar(1.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k', label='RF16 (Chowell et al.)')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([0.8-barWidth/2, 1.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(1.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'AUPRC_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(1.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(2.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([1.8-barWidth/2, 2.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(2.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'Accuracy_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(2.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(3.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([2.8-barWidth/2, 3.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(3.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'F1_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(3.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(4.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([3.8-barWidth/2, 4.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(4.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'MCC_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(4.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(5.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([4.8-barWidth/2, 5.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(5.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'BA_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(5.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(6.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([5.8-barWidth/2, 6.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(6.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    x_str = 'Performance_'+plot_train_test_delta
    y1 = performance_df.loc[performance_df['method']==MLM_list[0],x_str]
    y2 = performance_df.loc[performance_df['method']==MLM_list[1],x_str]
    ax.bar(6.8, np.mean(y1),
           yerr=np.std(y1), color=palette[0], capsize=5,
           width=barWidth, edgecolor='k')
    ax.bar(7.2, np.mean(y2),
           yerr=np.std(y2), color=palette[2], capsize=5,
           width=barWidth, edgecolor='k')
    u, w_pval = stats.mannwhitneyu(y1, y2)
    sig_str = sig_str_calculator(y1, y2, w_pval, sig_direction)
    print('%s %.3f %.3f %.3f %.3f %g'%(x_str, np.mean(y1), np.std(y1), np.mean(y2), np.std(y2), w_pval))
    if sig_str:
        ax.plot([6.8-barWidth/2, 7.2+barWidth/2], [ymax+0.05, ymax+0.05], color='k', linestyle='-', linewidth=0.5)
        ax.text(7.0, ymax + 0.06, sig_str, fontsize=8, color='k', horizontalalignment='center')


    ax.legend(frameon=False, loc=(0.1,0.95), prop={'size': 8},handlelength=1,handletextpad=0.1, labelspacing = 0.2, ncol=2)

    ax.set_xlim([0.4, 7.6])
    ax.set_xticks(range(1, 8))
    if plot_train_test_delta in ['train', 'test']:
        ax.set_ylim([0, 1])
        ax.set_xticklabels(['AUC', 'AUPRC', 'Accuracy', 'F1', 'MCC', 'BA', 'Performance'], rotation=45)
    else:
        ax.set_ylim([0, 0.5])
        ax.set_xticklabels(['\u0394 AUC', '\u0394 AUPRC', '\u0394 Accuracy', '\u0394 F1', '\u0394 MCC', '\u0394 BA', '\u0394 Performance'], rotation=45)
    ax.set_xlabel('')
    ax.set_ylabel('Value')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    fnOut = '../03.Results/PanCancer_LLR6_RF16NBT_Performance_compare_'+plot_train_test_delta+'.pdf'
    plt.savefig(fnOut)
