###############################################################################################
#Aim: Correlation between continuous features
#Description: To make heatmap showing correlation between continuous features (Fig. 1b).
#
#Run command: python 03.PanCancer_continuousFeatureCorrelation_heatmap.py
###############################################################################################


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from statsmodels.sandbox.stats.multicomp import multipletests
import copy

fontSize  = 8
plt.rcParams['font.size'] = fontSize
plt.rcParams["font.family"] = "Arial"

print('Raw data read in ...')
data_survival_fn = '../02.Input/features_phenotype_allDatasets.xlsx'
data_survival_Train = pd.read_excel(data_survival_fn, sheet_name='Chowell2015-2017', index_col=0)
data_survival_Test1 = pd.read_excel(data_survival_fn, sheet_name='Chowell2018', index_col=0)
data_survival_Test2 = pd.read_excel(data_survival_fn, sheet_name='Morris_new', index_col=0)
data_survival_Test3 = pd.read_excel(data_survival_fn, sheet_name='Morris_new2', index_col=0)
data_survival_Test4 = pd.read_excel(data_survival_fn, sheet_name='Lee_NSCLC', index_col=0)
data_survival_Test5 = pd.read_excel(data_survival_fn, sheet_name='Kurzrock_panCancer', index_col=0)
data_survival_Test6 = pd.read_excel(data_survival_fn, sheet_name='Vanguri_NSCLC_all', index_col=0)
data_survival_Test7 = pd.read_excel(data_survival_fn, sheet_name='Ravi_NSCLC', index_col=0)
data_survival_Test8 = pd.read_excel(data_survival_fn, sheet_name='Pradat_panCancer', index_col=0)
data_all_raw = pd.concat([data_survival_Train,data_survival_Test1,data_survival_Test2,data_survival_Test3,
                          data_survival_Test4,data_survival_Test5,data_survival_Test6,data_survival_Test7,
                          data_survival_Test8], axis=0) # data_survival_Test8,

all_features = ['CancerType', 'TMB', 'PDL1_TPS(%)', 'Chemo_before_IO', 'Albumin', 'NLR', 'Age', 'FCNA','Drug', 'Sex', 'MSI', 'Stage',
       'HLA_LOH', 'HED', 'Platelets', 'HGB', 'BMI', 'Pack_years', 'Smoking_status', 'Histology', 'Performance_status',
       'PFS_Event', 'PFS_Months', 'OS_Event', 'OS_Months', 'Response']
data_all = data_all_raw[all_features]

################ Correlation matrix heatmap plot between continuous variables that are present in >= 2 datasets ################
continuous_features = ['TMB', 'PDL1_TPS(%)', 'FCNA','HED', 'BMI', 'HGB', 'Albumin', 'NLR', 'Platelets', 'Age']
data_continuous_features = data_all_raw[continuous_features]

corr_out = data_continuous_features.corr(method='spearman', min_periods=1)
# set column and row names
corr_out.columns = data_continuous_features.columns
corr_out.index = data_continuous_features.columns
# create a mask to only show the lower triangle
mask = np.zeros_like(corr_out)
mask[np.triu_indices_from(mask)] = True
# set heatmap color palette and breaks
palette_length = 400
my_color = sns.color_palette("RdBu_r", n_colors=palette_length)
#my_color = plt.cm.cividis(range(palette_length))
#my_color = my_color.tolist()
#print(my_color)

corr_out.to_csv('../03.Results/source_data_fig01b_scc.csv', index=False)

# plot correlation heatmap
fig, ax = plt.subplots(figsize=(3.5, 3.2))
plt.subplots_adjust(left= 0.02, bottom=0.02, right=0.9, top=0.95)

heatmap = sns.heatmap(corr_out, mask=mask, cmap=my_color, center=0,
            vmin=-1, vmax=1, xticklabels=True, yticklabels=True,
            cbar=True, cbar_kws={"shrink": 0.5, "label": "Spearman correlation coefficient"}, cbar_ax=ax.inset_axes([0.72, 0.5, 0.04, 0.5]),
            linewidths=0.1, linecolor='white', square=True, ax=ax) # annot=True, fmt='.2f', annot_kws={"fontsize":7},

# Define significance levels
sig_levels = [(0.001, '***'), (0.01, '**'), (0.05, '*')]
# calculate significance symbols
p_list = []
for i in range(corr_out.shape[1]):
    for j in range(corr_out.shape[1]-1,i,-1):
        if mask[i, j]:
            corr, pval = spearmanr(data_continuous_features.iloc[:,i], data_continuous_features.iloc[:,j], nan_policy='omit')
            p_list.append(pval)
# adjusting p-values with multiple tests
adjusted_p_values = multipletests(p_list, method='bonferroni')[1] # bonferroni   fdr_bh
# add significance symbols
adjusted_p_values_out = copy.deepcopy(corr_out)
adjusted_p_values_out.loc[:, :] = ""
count = 0
for i in range(corr_out.shape[1]):
    for j in range(corr_out.shape[1]-1,i,-1):
        if mask[i, j]:
            adjusted_p_values_out.iloc[i, j] = adjusted_p_values[count]
            anno_text = '%.2f' % corr_out.iloc[i,j]
            adj_pval = adjusted_p_values[count]
            count+=1
            for level in sig_levels:
                if adj_pval < level[0]:
                    anno_text = '%.2f\n%s' % (corr_out.iloc[i,j], level[1])
                    break
            ax.text(i + 0.5, j + 0.5, anno_text, ha='center', va='center', fontsize=7, color='k')
cbar = heatmap.collections[0].colorbar
# Set the font size and font color for the colorbar
cbar.ax.tick_params(labelsize=8)

adjusted_p_values_out.to_csv('../03.Results/source_data_fig01b_adj_pval.csv', index=False)

# display the column names at the diagonal
continuous_features_full = ['TMB', 'PD-L1 TPS', 'FCNA','HED', 'BMI', 'Hemoglobin', 'Albumin', 'NLR', 'Platelets', 'Age'] # 'White blood cells', 'Neutrophils', 'Lymphocytes','Monocytes', 'Eosinophils', 'Smoking'
for i in range(len(corr_out.columns)):
    plt.text(i + 0.5, i + 0.5, continuous_features_full[i], ha='left', va='bottom', rotation=45, fontsize=8)
# show the plot
plt.xticks([])
plt.yticks([])
output_fig = '../03.Results/corHeatmap_pancancer.pdf' # png
plt.savefig(output_fig, transparent = True) # , dpi=300
plt.close()
