###############################################################################################
#Aim: Formula calculating LORIS
#Description: give the formula to calculate pan-cancer or NSCLC-specific LORIS based on feature values
#
#Run command, e.g.: python 12.Formula_LORIS.py LLR6_pan
###############################################################################################

import sys
import numpy as np

if __name__ == "__main__":
    model = sys.argv[1] # 'LLR6_pan'  'LLR6_NSCLC'

    ########################## Order of features ##########################
    if model == "LLR6_pan":
        featuresNA = ['TMB', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'CancerType1',
                      'CancerType2', 'CancerType3', 'CancerType4', 'CancerType5', 'CancerType6', 'CancerType7',
                      'CancerType8', 'CancerType9', 'CancerType10', 'CancerType11', 'CancerType12', 'CancerType13',
                      'CancerType14', 'CancerType15', 'CancerType16'] # pan-cancer feature order
    else:
        featuresNA = ['TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age'] # NSCLC feature order

    ###################### Read in LLRx model params ######################
    if model == 'LLR6_pan':
        fnIn = '../03.Results/6features/PanCancer/PanCancer_LLR6_10k_ParamCalculate.txt'
    elif model == 'LLR6_NSCLC':
        fnIn = '../03.Results/6features/NSCLC/NSCLC_LLR6_10k_ParamCalculate.txt'
    params_data = open(fnIn, 'r').readlines()
    params_dict = {}
    for line in params_data:
        if 'LLR_' not in line:
            continue
        words = line.strip().split('\t')
        param_name = words[0]
        params_val = [float(c) for c in words[1:]]
        params_dict[param_name] = params_val

    ########################## calculate LORIS formula ##########################
    scaler_mean_ = np.array(params_dict['LLR_mean'])
    scaler_scale_ = np.array(params_dict['LLR_scale'])
    clf_coef_ = np.array([params_dict['LLR_coef']])
    clf_intercept_ = np.array(params_dict['LLR_intercept'])

    coef_list = params_dict['LLR_coef']/scaler_scale_
    coef_list = [round(c,4) for c in coef_list]
    coef_tuple = list(zip(featuresNA, coef_list))
    print('Coef.: ', coef_tuple)

    interc = -sum(params_dict['LLR_coef']*scaler_mean_/scaler_scale_) + params_dict['LLR_intercept'][0]
    print('Intercept: ', round(interc,4))
