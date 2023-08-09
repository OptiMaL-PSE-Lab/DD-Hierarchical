
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dir_data = './results/Models/dataframe.csv'
save_fig = './results/Figures/correlation'

plt.rcParams["font.family"] = "Times New Roman"
ft = int(15)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 12.5,
              'legend.handlelength': 2}
plt.rcParams.update(params)

# plt.style.use('ggplot')

try:
    dataframe = pd.read_csv(dir_data)
except:
    print('Could not load data - Run save_pareto_data.py first')

df_small_sch = dataframe.copy().dropna(subset=['time_s']).reset_index(drop=True)
keys = ["test_score", "time_s", "time_l", "upper_s", "upper_l", "bi_s", "bi_l", "tri_s", "tri_l", "real"]
keys_lower = ["test_score", "time_s", "upper_s", "bi_s", "tri_s", "real"]

f = plt.figure(figsize=(10, 8))
plt.matshow(df_small_sch[keys].corr())
plt.xticks(range(len(keys)), keys, fontsize=11, rotation=45)
plt.yticks(range(len(keys)), keys, fontsize=11)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=11)
plt.title('Correlation Matrix unfiltered', fontsize=16)
plt.savefig(save_fig+'_unfiltered.svg')

f = plt.figure(figsize=(10, 8))
plt.matshow(df_small_sch[df_small_sch['type']=='schedu'][keys].corr())
plt.xticks(range(len(keys)), keys, fontsize=11, rotation=45)
plt.yticks(range(len(keys)), keys, fontsize=11)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=11)
plt.title('Correlation Matrix scheduling', fontsize=16)
plt.savefig(save_fig+'_scheduling.svg')

f = plt.figure(figsize=(10, 8))
plt.matshow(df_small_sch[df_small_sch['type']=='integr'][keys].corr())
plt.xticks(range(len(keys)), keys, fontsize=11, rotation=45)
plt.yticks(range(len(keys)), keys, fontsize=11)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=11)
plt.title('Correlation Matrix scheduling-control', fontsize=16)
plt.savefig(save_fig+'_integrated.svg')

fig, axs = plt.subplots(len(keys), len(keys), figsize=(20, 12))

for i,k1 in enumerate(list(keys)):
    for j, k2 in enumerate(list(keys)):
        dummy = df_small_sch[(df_small_sch[k1]!=None) & (df_small_sch[k2]!=None)]
        color = [(int(dummy['type'][i]=='schedu') , 0.6, 0) for i in range(len(dummy))]
        axs[i,j].scatter(dummy[k2], dummy[k1], c=color, s=11)
        if j==0:
            axs[i,j].set_ylabel(k1)
        if i == len(keys)-1:
            axs[i,j].set_xlabel(k2)
        # axs[i,j].set_xscale('log')
        # axs[i,j].set_yscale('log')
fig.show()
fig.savefig('./results/Figures/pairwise_plots.svg')


# X = df_small_sch[keys_lower].values
# scaler = StandardScaler()
# scaler.fit(X)
# X_norm = scaler.transform(X)
# pca = PCA(n_components=3) # estimate only 2 PCs
# X_new = pca.fit_transform(X) # project the original data into the PCA 
# print(pca.explained_variance_ratio_, )
# print(pca.components_, )








