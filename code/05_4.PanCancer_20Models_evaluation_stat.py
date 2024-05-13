###############################################################################################
#Aim: Machine learning model evaluation result
#Description: Get statistical result of machine learning model performance from the 2000-repeated 5-fold cross
# validation (Supplementary Tables 1,2).
#
#Run command: python 05_4.PanCancer_20Models_evaluation_stat.py
###############################################################################################


import pandas as pd
from scipy.stats import mannwhitneyu


MLM_list = ['LLR6', 'LR5noTMB', 'TMB', 'LogisticRegression', 'DecisionTree','RF6', 'RF16_NBT',
             'RandomForest', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost','LightGBM', 'SupportVectorMachineRadial',
             'kNearestNeighbourhood', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4',
             'GaussianProcess', "TabNet"]
MLM_list1=['TMB', 'DecisionTree', 'RF6', 'RF16_NBT', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes',
           'GaussianNaiveBayes', 'BernoulliNaiveBayes', 'RF16_NBT'] # data scaling: None
MLM_list2=['LLR6', 'LR5noTMB', 'LLR5noPSTH', 'LogisticRegression','GBoost', 'AdaBoost', 'HGBoost', 'XGBoost',
           'CatBoost', 'LightGBM','SupportVectorMachineLinear','SupportVectorMachinePoly','SupportVectorMachineRadial',
           'kNearestNeighbourhood','NeuralNetwork1','NeuralNetwork2','NeuralNetwork3','NeuralNetwork4',
           'GaussianProcess','QuadraticDiscriminantAnalysis', "TabNet"] # StandardScaler
MLM_name_map = {'TMB':'TMB',
                'LLR6': 'LLR6', 'LR5noTMB': 'LR5 (noTMB)', 'LLR5noPSTH': 'LLR5 (noHistory)', 'LogisticRegression': 'LR16',
                'DecisionTree':'DecisionTree','RF6':'RF6','RF16_NBT':'RF16 (Chowell et al.)',
                'RandomForest':'RandomForest','XGBoost':'XGBoost',
                'NeuralNetwork1':'MultilayerPerceptron (1 layer)','NeuralNetwork2':'MultilayerPerceptron (2 layers)',
                'NeuralNetwork3':'MultilayerPerceptron (3 layers)', 'NeuralNetwork4':'MultilayerPerceptron (4 layers)',
                'LightGBM':'LightGBM', 'GaussianProcess':'GaussianProcess', 'GBoost':'GBoost',
                'HGBoost':'HGBoost', 'AdaBoost':'AdaBoost', 'SupportVectorMachineRadial':'SupportVectorMachine',
                'kNearestNeighbourhood':'kNearestNeighbors', "TabNet":"TabNet"}

randomSeed = 1
N_repeat_KFold = 2000
Kfold = 5
resampleNUM = N_repeat_KFold*Kfold

performance_df = pd.DataFrame()

print('Raw data reading in ...')
##### 6-feature and 16-feature models
for MLM in MLM_list:
    print(MLM, ' in processing ...')
    temp_df = pd.DataFrame()
    temp_df['method'] = [MLM_name_map[MLM]] * resampleNUM
    if MLM in MLM_list1:
        SCALE = 'None'
    elif MLM in MLM_list2:
        SCALE = 'StandardScaler'
    else:
        raise Exception('MLM not recognized!')
    if MLM in ['LLR6', 'LR5noTMB', 'RF6', 'TMB']:
        fnIn = '../03.Results/6features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(
            Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    elif MLM in ["TabNet"]:
        fnIn = '../03.Results/16features/PanCancer/' + MLM + '_evaluation_2000R5CV_result.txt'
    else:
        fnIn = '../03.Results/16features/PanCancer/ModelEvalResult_' + MLM + '_Scaler(' + SCALE + ')_CV' + str(
            Kfold) + 'Rep' + str(N_repeat_KFold) + '_random' + str(randomSeed) + '.txt'
    if MLM not in ["TabNet"]:
        dataIn = open(fnIn, 'r').readlines()
        header = dataIn[0].strip().split('\t')
        position = header.index('params')
        data = dataIn[1].strip().split('\t')
        data = data[(position+2):]
        data = [float(c) for c in data]
        temp_df['AUC_test'] = data[resampleNUM*0:resampleNUM*1]
        temp_df['AUC_train'] = data[(resampleNUM * 1+3):(resampleNUM*2 + 3)]
        temp_df['PRAUC_test'] = data[(resampleNUM*2 + 3+2):(resampleNUM*3 + 3+2)]
        temp_df['PRAUC_train'] = data[(resampleNUM * 3 + 3*2 + 2):(resampleNUM * 4 + 3*2 + 2)]
        temp_df['Accuracy_test'] = data[(resampleNUM * 4 + 3*2 + 2*2):(resampleNUM * 5 + 3*2 + 2*2)]
        temp_df['Accuracy_train'] = data[(resampleNUM * 5 + 3*3 + 2*2):(resampleNUM * 6 + 3*3 + 2*2)]
        temp_df['F1_test'] = data[(resampleNUM * 6 + 3*3 + 2*3):(resampleNUM * 7 + 3*3 + 2*3)]
        temp_df['F1_train'] = data[(resampleNUM * 7 + 3*4 + 2*3):(resampleNUM * 8 + 3*4 + 2*3)]
        temp_df['MCC_test'] = data[(resampleNUM * 8 + 3 * 4 + 2 * 4):(resampleNUM * 9 + 3 * 4 + 2 * 4)]
        temp_df['MCC_train'] = data[(resampleNUM * 9 + 3 * 5 + 2 * 4):(resampleNUM * 10 + 3 * 5 + 2 * 4)]
        temp_df['BA_test'] = data[(resampleNUM * 10 + 3 * 5 + 2 * 5):(resampleNUM * 11 + 3 * 5 + 2 * 5)]
        temp_df['BA_train'] = data[(resampleNUM * 11 + 3 * 6 + 2 * 5):(resampleNUM * 12 + 3 * 6 + 2 * 5)]
        temp_df['Performance_test'] = (temp_df['AUC_test'] * temp_df['PRAUC_test'] * temp_df['Accuracy_test'] * temp_df[
            'F1_test']) ** (1 / 4)
        temp_df['Performance_train'] = (temp_df['AUC_train'] * temp_df['PRAUC_train'] * temp_df['Accuracy_train'] * temp_df[
            'F1_train']) ** (1 / 4)
        temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
        temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
        temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
        temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
        temp_df['MCC_delta'] = temp_df['MCC_train'] - temp_df['MCC_test']
        temp_df['BA_delta'] = temp_df['BA_train'] - temp_df['BA_test']
        temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
    else:
        temp_df = pd.read_csv(fnIn, header=None, sep='\t')
        temp_df.columns = ['AUC_train','PRAUC_train','F1_train','Accuracy_train','MCC_train','BA_train',
                          'AUC_test','PRAUC_test','F1_test','Accuracy_test','MCC_test','BA_test']
        temp_df['method'] = MLM_name_map[MLM]
        temp_df['Performance_test'] = (temp_df['AUC_test'] * temp_df['PRAUC_test'] * temp_df['Accuracy_test'] *
                                      temp_df['F1_test']) ** (1 / 4)
        temp_df['Performance_train'] = (temp_df['AUC_train'] * temp_df['PRAUC_train'] * temp_df['Accuracy_train'] *
                                       temp_df['F1_train']) ** (1 / 4)
        temp_df['AUC_delta'] = temp_df['AUC_train'] - temp_df['AUC_test']
        temp_df['PRAUC_delta'] = temp_df['PRAUC_train'] - temp_df['PRAUC_test']
        temp_df['Accuracy_delta'] = temp_df['Accuracy_train'] - temp_df['Accuracy_test']
        temp_df['F1_delta'] = temp_df['F1_train'] - temp_df['F1_test']
        temp_df['MCC_delta'] = temp_df['MCC_train'] - temp_df['MCC_test']
        temp_df['BA_delta'] = temp_df['BA_train'] - temp_df['BA_test']
        temp_df['Performance_delta'] = temp_df['Performance_train'] - temp_df['Performance_test']
        ordered_columns = ['method','AUC_test','AUC_train','PRAUC_test','PRAUC_train','Accuracy_test','Accuracy_train',
                           'F1_test','F1_train','MCC_test','MCC_train','BA_test','BA_train','Performance_test','Performance_train',
                           'AUC_delta','PRAUC_delta','Accuracy_delta','F1_delta','MCC_delta','BA_delta','Performance_delta']
        temp_df = temp_df[ordered_columns]
    performance_df = pd.concat([performance_df, temp_df], axis=0)

##### write to file
fnOut_test = '../03.Results/PanCancer_20Models_Performance_test.xlsx'
fnOut_delta = '../03.Results/PanCancer_20Models_Performance_delta.xlsx'
grouped = performance_df.groupby('method').agg({
    'AUC_test': ['mean', 'std'],
    'AUC_delta': ['mean', 'std'],
    'PRAUC_test': ['mean', 'std'],
    'PRAUC_delta': ['mean', 'std'],
    'Accuracy_test': ['mean', 'std'],
    'Accuracy_delta': ['mean', 'std'],
    'F1_test': ['mean', 'std'],
    'F1_delta': ['mean', 'std'],
    'MCC_test': ['mean', 'std'],
    'MCC_delta': ['mean', 'std'],
    'BA_test': ['mean', 'std'],
    'BA_delta': ['mean', 'std'],
    'Performance_test': ['mean', 'std'],
    'Performance_delta': ['mean', 'std']
}).reset_index()
# Rename the columns
grouped.columns = ['method', 'mean_AUC_test', 'std_AUC_test', 'mean_AUC_delta', 'std_AUC_delta',
                   'mean_PRAUC_test', 'std_PRAUC_test', 'mean_PRAUC_delta', 'std_PRAUC_delta',
                   'mean_Accuracy_test', 'std_Accuracy_test', 'mean_Accuracy_delta', 'std_Accuracy_delta',
                   'mean_F1_test', 'std_F1_test', 'mean_F1_delta', 'std_F1_delta',
                   'mean_MCC_test', 'std_MCC_test', 'mean_MCC_delta', 'std_MCC_delta',
                   'mean_BA_test', 'std_BA_test', 'mean_BA_delta', 'std_BA_delta',
                   'mean_Performance_test', 'std_Performance_test', 'mean_Performance_delta', 'std_Performance_delta']
pval_test = []
pval_delta = []
for MLM in grouped['method']:
    x1 = performance_df.loc[performance_df['method']=='LLR6','Performance_test']
    x2 = performance_df.loc[performance_df['method'] == MLM, 'Performance_test']
    u, w_pval = mannwhitneyu(x1, x2)
    pval_test.append(w_pval)
    x1 = performance_df.loc[performance_df['method'] == 'LLR6', 'Performance_delta']
    x2 = performance_df.loc[performance_df['method'] == MLM, 'Performance_delta']
    u, w_pval = mannwhitneyu(x1, x2)
    pval_delta.append(w_pval)

grouped['Rank_test'] = grouped['mean_Performance_test'].rank(ascending=False)
grouped['Rank_delta'] = grouped['mean_Performance_delta'].rank(ascending=True)
grouped = grouped.round(2)
grouped['pval_test'] = pval_test
grouped['pval_delta'] = pval_delta

content_df = pd.DataFrame()
content_df['method'] = grouped['method']
content_df['AUC'] = grouped['mean_AUC_test'].astype(str) + '±' + grouped['std_AUC_test'].astype(str)
content_df['PRAUC'] = grouped['mean_PRAUC_test'].astype(str) + '±' + grouped['std_PRAUC_test'].astype(str)
content_df['Accuracy'] = grouped['mean_Accuracy_test'].astype(str) + '±' + grouped['std_Accuracy_test'].astype(str)
content_df['F1'] = grouped['mean_F1_test'].astype(str) + '±' + grouped['std_F1_test'].astype(str)
content_df['MCC'] = grouped['mean_MCC_test'].astype(str) + '±' + grouped['std_MCC_test'].astype(str)
content_df['BA'] = grouped['mean_BA_test'].astype(str) + '±' + grouped['std_BA_test'].astype(str)
content_df['Performance'] = grouped['mean_Performance_test'].astype(str) + '±' + grouped['std_Performance_test'].astype(str)
content_df['Rank_test'] = grouped['Rank_test']
content_df['pval_test'] = grouped['pval_test']
sheet_name = 'Test'
content_df.to_excel(fnOut_test, sheet_name=sheet_name, index=False)

content_df = pd.DataFrame()
content_df['method'] = grouped['method']
content_df['AUC'] = grouped['mean_AUC_delta'].astype(str) + '±' + grouped['std_AUC_delta'].astype(str)
content_df['PRAUC'] = grouped['mean_PRAUC_delta'].astype(str) + '±' + grouped['std_PRAUC_delta'].astype(str)
content_df['Accuracy'] = grouped['mean_Accuracy_delta'].astype(str) + '±' + grouped['std_Accuracy_delta'].astype(str)
content_df['F1'] = grouped['mean_F1_delta'].astype(str) + '±' + grouped['std_F1_delta'].astype(str)
content_df['MCC'] = grouped['mean_MCC_delta'].astype(str) + '±' + grouped['std_MCC_delta'].astype(str)
content_df['BA'] = grouped['mean_BA_delta'].astype(str) + '±' + grouped['std_BA_delta'].astype(str)
content_df['Performance'] = grouped['mean_Performance_delta'].astype(str) + '±' + grouped['std_Performance_delta'].astype(str)
content_df['Rank_delta'] = grouped['Rank_delta']
content_df['pval_delta'] = grouped['pval_delta']
sheet_name = 'Delta'
content_df.to_excel(fnOut_delta, sheet_name=sheet_name, index=False)
