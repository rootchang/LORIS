###############################################################################################
#Aim: Score vs. ICB objective response odds
#Description: Relationship between LORIS or TMB score vs. ICB objective response odds
#             1) on all patients
#             2) on non-NSCLC patients
#             (Fig. 4a,b; Extended Data Fig. 8a,b).
#Run command, e.g.: python 07_1.PanCancer_LORIS_TMB_vs_resProb_curve.py PanCancer_all
###############################################################################################


import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import seaborn as sns

import sklearn.neighbors._base
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base

plt.rcParams.update({'font.size': 9})
plt.rcParams["font.family"] = "Arial"
palette = sns.color_palette("deep")


if __name__ == "__main__":

    bs_number = 1000 # multi-time resampling for mean,sd,CI
    random.seed(1)

    Plot_type = sys.argv[1] # 'PanCancer_all'  'PanCancer_nonNSCLC'
    bin_size = 0.1 # 0.05, 0.1, 0.15, 0.2

    start_time = time.time()
    print('Raw data read in ...')
    fnIn = '../03.Results/' +Plot_type+ '_LLR6_Scaler(StandardScaler)_prediction.xlsx'
    y_pred_LLR6 = []
    y_true = []
    start_set = 1
    end_set = 3
    output_curve_fn = '../03.Results/LLR6_LORIS_vs_ORR_'+Plot_type+'.png'
    for sheet_i in range(start_set,end_set): # range(start_set,3)
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_LLR6.extend(data['y_pred'].tolist())
        y_true.extend(data['y'].tolist())

    fnIn = '../03.Results/' + 'TMB_Scaler(None)_prediction.xlsx'
    y_pred_TMB = []
    for sheet_i in range(start_set,end_set):
        data = pd.read_excel(fnIn, sheet_name=str(sheet_i), header=0, index_col=0)
        y_pred_TMB.extend(data['y_pred'].tolist())

    y_true = np.array(y_true)
    y_pred_LLR6 = np.array(y_pred_LLR6)
    y_pred_TMB = np.array(y_pred_TMB)
    score_list_LLR6 = np.arange(0.0, 1.01, 0.01)  # 0.0, 0.92, 0.01
    score_list_TMB = np.arange(0, 101, 1)
    LLR6_num = len(score_list_LLR6)
    TMB_num = len(score_list_TMB)

    LLR6high_ORR_list = [[] for _ in range(LLR6_num)]
    TMBhigh_ORR_list = [[] for _ in range(TMB_num)]
    LLR6_ORR_list = [[] for _ in range(LLR6_num)]
    TMB_ORR_list = [[] for _ in range(TMB_num)]
    LLR6low_ORR_list = [[] for _ in range(LLR6_num)]
    TMBlow_ORR_list = [[] for _ in range(TMB_num)]

    LLR6_patientNUM_list = [[] for _ in range(LLR6_num)]
    TMB_patientNUM_list = [[] for _ in range(TMB_num)]
    sampleNUM = len(y_true)
    idx_list = range(sampleNUM)
    print('Sample num:',sampleNUM)

    for bs in range(bs_number):
        idx_resampled = random.choices(idx_list, k = sampleNUM)
        y_true_resampled = y_true[idx_resampled]
        y_pred_LLR6_resampled = y_pred_LLR6[idx_resampled]
        y_pred_TMB_resampled = y_pred_TMB[idx_resampled]

        for score_i in range(len(score_list_LLR6)):
            score = score_list_LLR6[score_i]
            idx_high_interval = y_pred_LLR6_resampled >= score
            y_true_high = y_true_resampled[idx_high_interval]
            Rhigh_num = sum(y_true_high)
            tot_high_num = len(y_true_high)
            patientRatio_temp = sum(y_pred_LLR6_resampled < score) / sampleNUM
            LLR6_patientNUM_list[score_i].append(patientRatio_temp)
            if not tot_high_num:
                LLR6high_ORR_list[score_i].append(LLR6high_ORR_list[score_i-1][-1])
            else:
                ORRhigh_temp = Rhigh_num / tot_high_num
                LLR6high_ORR_list[score_i].append(ORRhigh_temp)

            idx_low_interval = y_pred_LLR6_resampled < score
            y_true_low = y_true_resampled[idx_low_interval]
            Rlow_num = sum(y_true_low)
            tot_low_num = len(y_true_low)
            if not tot_low_num:
                LLR6low_ORR_list[score_i].append(0)
            else:
                ORRlow_temp = Rlow_num / tot_low_num
                LLR6low_ORR_list[score_i].append(ORRlow_temp)

            if sum(y_pred_LLR6_resampled <= score+bin_size/2) < 0.01*len(y_pred_LLR6_resampled): # skip
                idx_interval = []
            elif sum(y_pred_LLR6_resampled > score-bin_size/2) < 0.01*len(y_pred_LLR6_resampled): # merge patients
                idx_interval = (y_pred_LLR6_resampled > score-bin_size/2)
            else:
                idx_interval = (y_pred_LLR6_resampled <= score+bin_size/2) & (y_pred_LLR6_resampled > score-bin_size/2)
            y_true_temp = y_true_resampled[idx_interval]
            R_num = sum(y_true_temp)
            tot_num = len(y_true_temp)
            if not tot_num:
                #1
                LLR6_ORR_list[score_i].append(0)
            else:
                ORR_temp = R_num / tot_num
                LLR6_ORR_list[score_i].append(ORR_temp)
            if sum(y_pred_LLR6_resampled > score - bin_size/2) < 0.01 * len(y_pred_LLR6_resampled):  # finish score_list_LLR6 loop
                break
        for score_i in range(len(score_list_TMB)):
            score = score_list_TMB[score_i]
            idx_high_interval = y_pred_TMB_resampled >= score
            y_true_high = y_true_resampled[idx_high_interval]
            Rhigh_num = sum(y_true_high)
            tot_high_num = len(y_true_high)
            patientRatio_temp = sum(y_pred_TMB_resampled < score) / sampleNUM
            TMB_patientNUM_list[score_i].append(patientRatio_temp)
            if not tot_high_num:
                TMBhigh_ORR_list[score_i].append(TMBhigh_ORR_list[score_i - 1][-1])
            else:
                ORRhigh_temp = Rhigh_num / tot_high_num
                TMBhigh_ORR_list[score_i].append(ORRhigh_temp)

            idx_low_interval = y_pred_TMB_resampled < score
            y_true_low = y_true_resampled[idx_low_interval]
            Rlow_num = sum(y_true_low)
            tot_low_num = len(y_true_low)
            if not tot_low_num:
                TMBlow_ORR_list[score_i].append(0)
            else:
                ORRlow_temp = Rlow_num / tot_low_num
                TMBlow_ORR_list[score_i].append(ORRlow_temp)

            if sum(y_pred_TMB_resampled <= score + 5) < 0.01 * len(y_pred_TMB_resampled):  # skip
                idx_interval = []
            elif sum(y_pred_TMB_resampled > score - 5) < 0.01 * len(y_pred_TMB_resampled):  # merge patients
                idx_interval = (y_pred_TMB_resampled > score - 5)
            else:
                idx_interval = (y_pred_TMB_resampled <= score + 5) & (y_pred_TMB_resampled > score - 5)
            y_true_temp = y_true_resampled[idx_interval]
            R_num = sum(y_true_temp)
            tot_num = len(y_true_temp)
            if not tot_num:
                TMB_ORR_list[score_i].append(0)
            else:
                ORR_temp = R_num / tot_num
                TMB_ORR_list[score_i].append(ORR_temp)
            if sum(y_pred_TMB_resampled > score - 5) < 0.01 * len(y_pred_TMB_resampled):  # finish score_list_LLR6 loop
                break
    # remove empty elements (upon scores that near 1)
    for i in range(len(LLR6high_ORR_list)):
        if len(LLR6high_ORR_list[i])==0:
            break
    LLR6high_ORR_list = LLR6high_ORR_list[0:i]
    LLR6low_ORR_list = LLR6low_ORR_list[0:i]
    LLR6_ORR_list = LLR6_ORR_list[0:i]
    print([len(c) for c in LLR6_ORR_list])
    LLR6_patientNUM_list = LLR6_patientNUM_list[0:i]
    score_list_LLR6 = score_list_LLR6[0:i]
    LLR6high_ORR_mean = [np.mean(c) for c in LLR6high_ORR_list]
    LLR6high_ORR_05 = [np.quantile(c,0.05) for c in LLR6high_ORR_list]
    LLR6high_ORR_95 = [np.quantile(c,0.95) for c in LLR6high_ORR_list]
    LLR6low_ORR_mean = [np.mean(c) for c in LLR6low_ORR_list]
    LLR6low_ORR_05 = [np.quantile(c, 0.05) for c in LLR6low_ORR_list]
    LLR6low_ORR_95 = [np.quantile(c, 0.95) for c in LLR6low_ORR_list]
    LLR6low_patientRatio_mean = [np.mean(c) for c in LLR6_patientNUM_list]
    LLR6_ORR_mean = [np.mean(c) for c in LLR6_ORR_list]
    LLR6_ORR_05 = [np.quantile(c, 0.05) for c in LLR6_ORR_list]
    LLR6_ORR_95 = [np.quantile(c, 0.95) for c in LLR6_ORR_list]
    LLR6_patientRatio_mean = [np.mean(c) for c in LLR6_patientNUM_list]
    print('LLR6 response odds:')
    for i in range(len(LLR6high_ORR_95)):
        print(score_list_LLR6[i], LLR6high_ORR_mean[i], LLR6low_ORR_mean[i], LLR6_ORR_mean[i], LLR6low_patientRatio_mean[i])

    # remove empty elements (upon scores that near 1)
    for i in range(len(TMBhigh_ORR_list)):
        if len(TMBhigh_ORR_list[i]) == 0:
            break
    TMBhigh_ORR_list = TMBhigh_ORR_list[0:i]
    TMBlow_ORR_list = TMBlow_ORR_list[0:i]
    TMB_ORR_list = TMB_ORR_list[0:i]
    TMB_patientNUM_list = TMB_patientNUM_list[0:i]
    score_list_TMB = score_list_TMB[0:i]
    TMBhigh_ORR_mean = [np.mean(c) for c in TMBhigh_ORR_list]
    TMBhigh_ORR_05 = [np.quantile(c, 0.05) for c in TMBhigh_ORR_list]
    TMBhigh_ORR_95 = [np.quantile(c, 0.95) for c in TMBhigh_ORR_list]
    TMBlow_ORR_mean = [np.mean(c) for c in TMBlow_ORR_list]
    TMBlow_ORR_05 = [np.quantile(c, 0.05) for c in TMBlow_ORR_list]
    TMBlow_ORR_95 = [np.quantile(c, 0.95) for c in TMBlow_ORR_list]
    TMBlow_patientRatio_mean = [np.mean(c) for c in TMB_patientNUM_list]
    TMB_ORR_mean = [np.mean(c) for c in TMB_ORR_list]
    TMB_ORR_05 = [np.quantile(c, 0.05) for c in TMB_ORR_list]
    TMB_ORR_95 = [np.quantile(c, 0.95) for c in TMB_ORR_list]
    TMB_patientRatio_mean = [np.mean(c) for c in TMB_patientNUM_list]
    print('TMB response odds:')
    for i in range(len(TMBhigh_ORR_95)):
        print(score_list_TMB[i], TMBhigh_ORR_mean[i], TMBlow_ORR_mean[i], TMB_ORR_mean[i], TMBlow_patientRatio_mean[i])

    df = pd.DataFrame({'LLR6_score': score_list_LLR6, 'Prob_mean': LLR6_ORR_mean, 'Prob_lower': LLR6_ORR_05, 'Prob_upper': LLR6_ORR_95})
    df.to_csv('../03.Results/source_data_fig05a.csv', index=False)
    df = pd.DataFrame({'TMB': score_list_TMB, 'Prob_mean': TMB_ORR_mean, 'Prob_lower': TMB_ORR_05, 'Prob_upper': TMB_ORR_95})
    df.to_csv('../03.Results/source_data_fig05b.csv', index=False)


    ############# Score-Prob curve ##############
    subplot_num = 2
    fig1, axes = plt.subplots(1, subplot_num, figsize=(6.5, 2.8))
    fig1.subplots_adjust(left=0.1, bottom=0.15, right=0.98, top=0.96, wspace=0.25, hspace=0.45)
    axes[0].plot(score_list_LLR6, LLR6_ORR_mean, '-', color='r')
    axes[0].fill_between(score_list_LLR6, LLR6_ORR_05, LLR6_ORR_95, facecolor='r', alpha=0.25)
    axes[0].set_ylabel("Response probability (%)", color="k")
    axes[0].set_xlabel('LORIS') # L'LLR6 score'

    axes[-1].plot(score_list_TMB, TMB_ORR_mean, '-', color='r')
    axes[-1].fill_between(score_list_TMB, TMB_ORR_05, TMB_ORR_95, facecolor='r', alpha=0.25)
    axes[-1].set_ylabel("Response probability (%)", color="k")
    axes[-1].set_xlabel('TMB')


    for j in range(subplot_num):
        axes[j].set_ylim([-0.02, 1.02])
        axes[j].set_yticks([0, 0.25, 0.5, 0.75, 1])
        axes[j].set_yticklabels([0, 25, 50, 75, 100])
        axes[j].spines['right'].set_visible(False)
        axes[j].spines['top'].set_visible(False)

    for j in range(subplot_num-1):
        axes[j].set_xlim([0, 1])
        axes[j].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        axes[j].set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    axes[-1].set_xlim([0, 53])
    axes[-1].set_xticks([0, 10, 20, 30,40,50])

    if Plot_type == 'PanCancer_all':
        axes[0].axvspan(0, 0.275, facecolor='k', alpha=0.1)
        axes[0].axvspan(0.695, 1, facecolor='g', alpha=0.1)
        axes[-1].axvspan(26.5, 100, facecolor='g', alpha=0.1)
    if Plot_type=='PanCancer_nonNSCLC':
        axes[0].axvspan(0, 0.275, facecolor='k', alpha=0.1)
        axes[0].axvspan(0.7, 1, facecolor='g', alpha=0.1)

    plt.savefig(output_curve_fn, dpi=600)
    plt.close()