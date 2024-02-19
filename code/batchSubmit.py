############################################################################################################
# Submit batch jobs to server for model hyperparameter search and/or evaluation.
############################################################################################################

import os
import sys


RAND = sys.argv[1] # 1
TASK = sys.argv[2] # 'PS'  'PE'  'NS'  'NE'

if TASK in ['PS', 'NS']:
    MLM_list = ['DecisionTree', 'RandomForest','LogisticRegression', 'GBoost', 'AdaBoost',
                'HGBoost', 'XGBoost', 'LightGBM', 'SupportVectorMachineRadial','kNearestNeighbourhood',
                 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3', 'NeuralNetwork4', 'GaussianProcess']
elif TASK == 'PE':
    MLM_list = ['TMB', 'LLR6', 'LLR5noTMB', 'RF6', 'RF16_NBT', 'DecisionTree', 'RandomForest',
                'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'LightGBM',
                'SupportVectorMachineRadial','kNearestNeighbourhood', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3',
                'NeuralNetwork4','GaussianProcess']
elif TASK == 'NE':
    MLM_list = ['TMB', 'LLR6', 'LLR5noTMB', 'RF6', 'RF16_NBT', 'DecisionTree', 'RandomForest',
                'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost', 'XGBoost', 'LightGBM',
                'SupportVectorMachineRadial','kNearestNeighbourhood', 'NeuralNetwork1', 'NeuralNetwork2', 'NeuralNetwork3',
                'NeuralNetwork4','GaussianProcess']

MLM_list1 = ['TMB', 'RF6', 'RF16_NBT', 'DecisionTree', 'RandomForest', 'ComplementNaiveBayes', 'MultinomialNaiveBayes', 'GaussianNaiveBayes',
             'BernoulliNaiveBayes']  # data scaling: None
MLM_list2 = ['LLR6', 'LLR5noTMB', 'LogisticRegression', 'GBoost', 'AdaBoost', 'HGBoost','XGBoost', 'CatBoost',
             'LightGBM', 'SupportVectorMachineRadial','kNearestNeighbourhood', 'NeuralNetwork1', 'NeuralNetwork2',
             'NeuralNetwork3', 'NeuralNetwork4','GaussianProcess']  # StandardScaler

################################################## PanCancer ##################################################
if TASK in ['PS', 'PE']:
    for method in MLM_list:
        if method in MLM_list1:
            data_scale = 'None'
        else:
            data_scale = 'StandardScaler'
        jobNA = method+'_'+data_scale+'_'+RAND+'_Pan.run'
        foutNA = 'slurm-'+method+'_'+data_scale+'_'+RAND+'_Pan.out'
        command = 'sbatch --job-name='+jobNA+' --output='+foutNA+' --export=TASK='+TASK+',MLM='+method+',RAND='+RAND+' jobscript.sh'
        print(command)
        os.system(command)

################################################## NSCLC ##################################################
if TASK in ['NS', 'NE']:
    for DATA in ['Chowell']:
        for method in MLM_list:
            if method in MLM_list1:
                data_scale = 'None'
            else:
                data_scale = 'StandardScaler'
            jobNA = method+'_'+data_scale+'_'+DATA+'_'+RAND+'_NSCLC.run'
            foutNA = 'slurm-'+method+'_'+data_scale+'_'+DATA+'_'+RAND+'_NSCLC.out'
            command = 'sbatch --job-name='+jobNA+' --output='+foutNA+' --export=TASK='+TASK+',MLM='+method+',DATA='+DATA+',RAND='+RAND+' jobscript.sh'
            print(command)
            os.system(command)
