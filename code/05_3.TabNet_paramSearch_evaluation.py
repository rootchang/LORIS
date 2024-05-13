###############################################################################################
#Aim: Hyper-parameter search or evaluate of TabNet
#Description: To search the optimal parameters for TabNet, or evaluate the performance of the best TabNet model on
#             pan-cancer data or NSCLC data.
#
#Run command, e.g.: python -W ignore 05_3.TabNet_paramSearch_evaluation.py -1 1 10000 0.8 1000 ps NSCLC
###############################################################################################


import sys
import copy
import scipy
import numpy as np
import time
import pandas as pd
from collections import Counter
from sklearn import metrics
from sklearn.model_selection import train_test_split,ParameterGrid,KFold,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler,LabelEncoder
import torch
from pytorch_tabnet.augmentations import ClassificationSMOTE
from pytorch_tabnet.tab_model import TabNetClassifier


def dataScaler(data, featuresNA, numeric_featuresNA, scaler_type):
    data_scaled = copy.deepcopy(data)
    if scaler_type == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler_type == 'MinMax':
        scaler = MinMaxScaler()
    else:
        raise Exception('Unrecognized scaler type of %s! Only "sd" and "mM" are accepted.' % scaler_type)
    for feature in numeric_featuresNA:
        data_scaled[feature] = scaler.fit_transform(data[[feature]])
    x = pd.DataFrame(data_scaled, columns=featuresNA)
    return x


def CVsetGenerator(randomSeed,dataChowell_Train,featuresNA,phenoNA):
    # Generate 5-fold CV splits
    kf = KFold(n_splits=5, shuffle=True, random_state=randomSeed)
    train_sets = []
    train_train_sets = []
    train_val_sets = []
    val_sets = []
    for train_index, val_index in kf.split(dataChowell_Train):
        X_train, X_val = dataChowell_Train.iloc[train_index], dataChowell_Train.iloc[val_index]
        train_sets.append(X_train)
        val_sets.append(X_val)
        X_train_train, X_train_val = train_test_split(X_train, test_size=0.2) # , random_state=42
        train_train_sets.append(X_train_train)
        train_val_sets.append(X_train_val)
    X_train_sets = []
    y_train_sets = []
    y_train_train_sets = []
    y_train_val_sets = []
    X_valid_sets = []
    y_valid_sets = []
    sparse_X_train_sets = []
    sparse_X_train_train_sets = []
    sparse_X_train_val_sets = []
    sparse_X_valid_sets = []
    for i in range(len(train_sets)):
        X_train_sets.append(train_sets[i][featuresNA])
        y_train_sets.append(train_sets[i][phenoNA])
        y_train_train_sets.append(train_train_sets[i][phenoNA])
        y_train_val_sets.append(train_val_sets[i][phenoNA])
        X_valid_sets.append(val_sets[i][featuresNA])
        y_valid_sets.append(val_sets[i][phenoNA])
        sparse_X_train_sets.append(scipy.sparse.csr_matrix(X_train_sets[-1]))
        sparse_X_train_train_sets.append(scipy.sparse.csr_matrix(train_train_sets[i][featuresNA]))
        sparse_X_train_val_sets.append(scipy.sparse.csr_matrix(train_val_sets[i][featuresNA]))
        sparse_X_valid_sets.append(scipy.sparse.csr_matrix(X_valid_sets[-1]))
    return y_train_sets,y_train_train_sets,y_train_val_sets,y_valid_sets,\
           sparse_X_train_sets,sparse_X_train_train_sets,sparse_X_train_val_sets,sparse_X_valid_sets


def performance_calculator(y_true, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
    specificity_sensitivity_sum = tpr + (1 - fpr)
    ind_max = np.argmax(specificity_sensitivity_sum)
    if ind_max < 0.5:  # the first threshold is larger than all x values (tpr=1, fpr=1)
        ind_max = 1
    opt_cutoff = thresholds[ind_max]
    print('LLR opt_cutoff: ', opt_cutoff)
    y_pred_01 = [int(c >= opt_cutoff) for c in y_pred]
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_01).ravel()

    auc = metrics.roc_auc_score(y_true, y_pred)
    pr_auc = metrics.average_precision_score(y_true, y_pred)
    PPV = tp / (tp + fp)
    Sensitivity = tp / (tp + fn)
    Specificity = tn / (tn + fp)
    f1_score = 2 * PPV * Sensitivity / (PPV + Sensitivity)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    MCC = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    balanced_accuracy = (Sensitivity + Specificity) / 2
    return auc,pr_auc,f1_score,accuracy,MCC,balanced_accuracy


if __name__ == "__main__":
    start_time = time.time()

    print('Raw data processing ...')
    CPU_num = int(sys.argv[1])  # -1
    randomSeed = int(sys.argv[2]) # 1
    resampleNUM = int(sys.argv[3]) # 10000
    train_size = float(sys.argv[4]) # 0.8
    train_sample_size_used = int(sys.argv[5]) # use how many samples to train the LLR6
    task = sys.argv[6] # ps: paramSearch, ev: evaluation
    cancer_type = sys.argv[7] # Pan or NSCLC

    phenoNA = 'Response'
    if cancer_type == 'Pan':
        featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI', 'Stage',
                      'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'CancerType']
        xy_colNAs = featuresNA + [phenoNA]
    elif cancer_type == 'NSCLC':
        featuresNA = ['TMB', 'Systemic_therapy_history', 'Albumin', 'FCNA', 'NLR', 'Age', 'Drug', 'Sex', 'MSI', 'Stage',
                      'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'PDL1_TPS(%)']
        xy_colNAs = featuresNA + [phenoNA] + ['CancerType']

    dataALL_fn = '../02.Input/AllData.xlsx'
    dataChowell_Train0 = pd.read_excel(dataALL_fn, sheet_name='Chowell_train', index_col=0)
    dataChowell_Train0 = dataChowell_Train0[xy_colNAs]
    dataChowell_Train = copy.deepcopy(dataChowell_Train0)
    if train_sample_size_used > dataChowell_Train0.shape[0]:
        train_sample_size_used = dataChowell_Train0.shape[0]
    train_ratio_used = min(1, train_sample_size_used / dataChowell_Train.shape[0])
    if train_ratio_used < 0.99999:
        dataChowell_Train, useless = train_test_split(dataChowell_Train, test_size=1-train_ratio_used,
                      random_state=randomSeed, stratify=dataChowell_Train['CancerType'])  # stratify=None

    if cancer_type == 'NSCLC':
        dataChowell_Train = dataChowell_Train.loc[dataChowell_Train['CancerType']=='NSCLC',]
    print(dataChowell_Train.shape)

    # truncate extreme values of features
    TMB_upper = 50
    Age_upper = 85
    NLR_upper = 25
    dataChowell_Train['TMB'] = [c if c < TMB_upper else TMB_upper for c in dataChowell_Train['TMB']]
    dataChowell_Train['Age'] = [c if c < Age_upper else Age_upper for c in dataChowell_Train['Age']]
    dataChowell_Train['NLR'] = [c if c < NLR_upper else NLR_upper for c in dataChowell_Train['NLR']]

    print('Chowell patient number (training): ', dataChowell_Train.shape[0])
    counter = Counter(dataChowell_Train[phenoNA])
    pos_weight = counter[0] / counter[1]
    print('  Phenotype name: ', phenoNA)
    print('  Negative/Positive samples in training set: ', pos_weight)

    nunique = dataChowell_Train[featuresNA].nunique()
    types = dataChowell_Train[featuresNA].dtypes
    categorical_columns = []
    categorical_dims =  {}
    for col in dataChowell_Train[featuresNA].columns:
        if types[col] == 'object' or nunique[col] < 18:
            print(col, dataChowell_Train[col].nunique())
            l_enc = LabelEncoder()
            dataChowell_Train[col] = dataChowell_Train[col].fillna("VV_likely")
            dataChowell_Train[col] = l_enc.fit_transform(dataChowell_Train[col].values)
            categorical_columns.append(col)
            categorical_dims[col] = len(l_enc.classes_)
        else:
            dataChowell_Train.fillna(dataChowell_Train[col].mean(), inplace=True)
    print(categorical_columns)
    numeric_featuresNA = list(set(featuresNA) - set(categorical_columns))
    dataChowell_Train = dataScaler(dataChowell_Train, xy_colNAs, numeric_featuresNA, 'StandardScaler')

    grouped_features = []
    cat_idxs = [ i for i, f in enumerate(featuresNA) if f in categorical_columns]
    cat_dims = [ categorical_dims[f] for i, f in enumerate(featuresNA) if f in categorical_columns]

    max_epochs = 50
    aug = ClassificationSMOTE(p=0.2)
    model = TabNetClassifier(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=2,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_params={"gamma": 0.95,
                          "step_size": 20},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type='entmax',  # "sparsemax"
        grouped_features=grouped_features,
        epsilon=1e-15,
        verbose=0
    )

    if task == 'ps': # paramSearch
        # Generate 5-fold CV splits
        y_train_sets, y_train_train_sets, y_train_val_sets, y_valid_sets, \
        sparse_X_train_sets, sparse_X_train_train_sets, sparse_X_train_val_sets, sparse_X_valid_sets = \
            CVsetGenerator(randomSeed, dataChowell_Train, featuresNA, phenoNA)
        # Generate the parameter grid.
        param_grid = dict(n_d=[24, 32],  # 3,6,12,
                          n_a=[24],
                          n_steps=[3, 4, 5],
                          gamma=[1, 1.5, 2],
                          lambda_sparse=[1e-2, 1e-3, 1e-4],
                          momentum=[0.3, 0.4, 0.5],
                          n_shared=[2],
                          n_independent=[2],
                          clip_value=[2.],
                          )
        grid = ParameterGrid(param_grid)

        params_best = ""
        AUC_max = 0
        params_ind = 0
        for params in grid:
            params['n_a'] = params['n_d']
            AUC_mean = 0
            for i in range(5): # 5-fold CV
                tabnet = model
                tabnet.set_params(**params)
                tabnet.fit(X_train=sparse_X_train_sets[i], y_train=y_train_sets[i],
                           eval_set=[(sparse_X_valid_sets[i], y_valid_sets[i])],
                           eval_name=['CV'],
                           eval_metric=['auc'],
                           max_epochs=max_epochs, patience=10,
                           batch_size=1024, virtual_batch_size=128,
                           num_workers=0,
                           weights=1,
                           drop_last=False,
                           augmentations=aug,
                           )
                # Calculate AUC
                y_test_pred = tabnet.predict_proba(sparse_X_valid_sets[i])[:, 1]
                auc_test = metrics.roc_auc_score(y_valid_sets[i], y_test_pred)
                AUC_mean = AUC_mean+auc_test/5
            params_ind+=1
            if auc_test > AUC_max:
                params_best = params
                AUC_max = auc_test
            print('AUC_max = %.3f, Params_best = %s'%(AUC_max, params_best))
        print('Done. Time used: %d' % (time.time() - start_time))

    elif task == 'ev': # paramSearch
        if cancer_type == 'NSCLC':
            fnOut = '../03.Results/16features/NSCLC/TabNet_evaluation_2000R5CV_result_NSCLC.txt'
            params = {'clip_value': 2.0, 'gamma': 2, 'lambda_sparse': 0.001, 'momentum': 0.3,
                      'n_a': 24, 'n_d': 24, 'n_independent': 2, 'n_shared': 2, 'n_steps': 3}
        elif cancer_type == 'Pan':
            fnOut = '../03.Results/16features/PanCancer/TabNet_evaluation_2000R5CV_result.txt'
            params = {'clip_value': 2.0, 'gamma': 1.5, 'lambda_sparse': 0.0001, 'momentum': 0.5, 'n_a': 32, 'n_d': 32,
                      'n_independent': 2, 'n_shared': 2, 'n_steps': 5}
        fhOut = open(fnOut,'w')
        for i0 in range(2000):  # 2000-repeated 5-fold CV
            print('Evaluating model, round %d...'%(i0+1))
            # Generate 5-fold CV splits
            y_train_sets, y_train_train_sets, y_train_val_sets, y_valid_sets, \
            sparse_X_train_sets, sparse_X_train_train_sets, sparse_X_train_val_sets, sparse_X_valid_sets = \
                CVsetGenerator(randomSeed*(i0+1), dataChowell_Train, featuresNA, phenoNA)
            for i in range(5):  # 5-fold CV
                tabnet = model
                tabnet.set_params(**params)
                tabnet.fit(X_train=sparse_X_train_train_sets[i], y_train=y_train_train_sets[i],
                           eval_set=[(sparse_X_train_val_sets[i], y_train_val_sets[i])],
                           eval_name=['CV'],
                           eval_metric=['auc'],
                           max_epochs=max_epochs, patience=10,
                           batch_size=1024, virtual_batch_size=128,
                           num_workers=0,
                           weights=1,
                           drop_last=False,
                           augmentations=aug, 
                           )
                # Calculate all different metrics on train and test data
                y_train_pred = tabnet.predict_proba(sparse_X_train_sets[i])[:, 1]
                auc_train, pr_auc_train, f1_score_train, accuracy_train, MCC_train, balanced_accuracy_train = \
                    performance_calculator(y_train_sets[i], y_train_pred)
                y_test_pred = tabnet.predict_proba(sparse_X_valid_sets[i])[:, 1]
                auc_test, pr_auc_test, f1_score_test, accuracy_test, MCC_test, balanced_accuracy_test = \
                    performance_calculator(y_valid_sets[i], y_test_pred)
                content = [auc_train, pr_auc_train, f1_score_train, accuracy_train, MCC_train, balanced_accuracy_train,
                           auc_test, pr_auc_test, f1_score_test, accuracy_test, MCC_test, balanced_accuracy_test]
                fhOut.write('\t'.join([str(c) for c in content])+'\n')
            fhOut.flush()
        fhOut.close()
        print('Done. Time used: %d'%(time.time()-start_time))

