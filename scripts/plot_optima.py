
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

dir_data = './results/Optima/optima.csv'
dir_data2 = './results/Optima/optima_surr.csv'
save_fig = './results/Figures/'


plt.rcParams["font.family"] = "Times New Roman"
ft = int(13)
font = {'size': ft}
plt.rc('font', **font)
params = {'legend.fontsize': 11,
              'legend.handlelength': 2}
plt.rcParams.update(params)

# plt.style.use('ggplot')
# plt.rcParams['axes.facecolor']='white'
# plt.rcParams['savefig.facecolor']='white'

try:
    dataframe = pd.read_csv(dir_data)
    dataframe_surr = pd.read_csv(dir_data2)
except:
    print('Could not load data - Run save_pareto_data.py first')

dataframe.sort_values('method')

# fig = plt.figure(figsize=(8, 6))
# ax = plt.subplot(111)
# sns.barplot(x="method", 
#     y="opt_time",
#     data=dataframe, )
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # # Put a legend to the right of the current axis
# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()
# plt.yscale('log')
# plt.ylabel('Online optimization time [s]')
# plt.xlabel(' ')
# plt.xticks(rotation=35, fontsize=12)
# fig.savefig(save_fig+'opt_time_all.svg')

###

# DFOdict = {
#     'time': [],
#     'type': [],
#     'method': [],
# }

methods = dataframe['method']

# for i in range(len(dataframe)):
#     if methods[i][:3]=='DFO' and methods[i][:7]!='DFO_tri':
#         DFOdict['time'] += [dataframe['opt_time'][i]]
#         if methods[i][:6]=='DFO_bi':
#             DFOdict['type'] += ['all'] if len(methods[i])==6 else [methods[i][7:]]
#             DFOdict['method'] += ['bi']
#         else:
#             DFOdict['type'] += ['all'] if len(methods[i])==10 else [methods[i][11:]]
#             DFOdict['method'] += ['approx']

# DFOframe = pd.DataFrame(DFOdict)

# # DFOframe.sort_values('type')

# fig = plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
# ax = plt.subplot(111)
# sns.barplot(x="type", 
#     y="time",
#     data=DFOframe,
#     hue='method',
#     order=['low', 'distr', 'all'] )
# ax.legend(fontsize=12)
# # box = ax.get_position()
# # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # # Put a legend to the right of the current axis
# # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()
# plt.yscale('log')
# plt.ylabel('Online optimization time [s]')
# plt.xlabel('Type')
# # plt.xticks(rotation=35, fontsize=8)
# # plt.tight_layout()
# # plt.gcf().subplots_adjust(bottom=0.3)
# fig.savefig(save_fig+'opt_time_DFO.svg')

###

all = [
    'planning', 
    'DFO_bi', 
    'DFO_approx', 
]



dataframe2 = dataframe[dataframe['method'].isin(all)]
# dataframe2.sort_values('method')

plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
sns.barplot(x="method", 
    y="opt_time",
    data=dataframe2,
    errorbar=None )
plt.yscale('log')
plt.ylabel('Online optimization time [s]')
# plt.xlabel('  ')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
# plt.style.use('ggplot')
plt.ylim(bottom=0.001)
plt.xlabel('  ')
plt.savefig(save_fig+'opt_time_DFO_2.svg')

to_idx = {
    method: i for i, method in enumerate(methods)
}

dummy_df = pd.DataFrame({'planning': [dataframe['opt_time'][to_idx['planning']],
                                        dataframe['opt_time'][to_idx['planning']],
                                         dataframe['opt_time'][to_idx['planning']]],
                         'DFO_bi': [dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi_low']],
                                dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi_distr']],
                                dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi']]],
                         'DFO_approx': [dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi_low']]+dataframe['opt_time'][to_idx['DFO_approx_low']],
                                    dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi_distr']]+dataframe['opt_time'][to_idx['DFO_approx_distr']],
                                    dataframe['opt_time'][to_idx['planning']]+dataframe['opt_time'][to_idx['DFO_bi']]+dataframe['opt_time'][to_idx['DFO_approx']]]},
                        index=['low', 'distr', 'all'])

fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

dummy_df.plot(kind='bar', stacked=True, ax=ax)
 
plt.legend(fontsize=12)
plt.ylabel('Online optimization time [s]')
plt.yscale('log')
# plt.xlabel('Type')
# plt.style.use('ggplot')
plt.ylim(bottom=0.001)
plt.xlabel('  ')
fig.savefig(save_fig+'opt_time_DFO_3.svg')

DFO2dict = {
    'eval': [],
    'type': [],
    'method': [],
}

eval_type = ['upper', 'bi', 'tri', 'real']
eval_alias = ['upper', 'bi', 'approx', 'tri']
best_DFO = ['planning', 'DFO_bi', 'DFO_approx']

for i in range(len(dataframe)):
    if methods[i] in best_DFO:
        DFO2dict['eval'] += [dataframe[type][i] for type in eval_type]
        DFO2dict['method'] += [methods[i]]*4
        DFO2dict['type'] += eval_alias

DFO2frame = pd.DataFrame(DFO2dict)

fig = plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
ax = plt.subplot(111)
sns.barplot(x="type", 
    y="eval",
    data=DFO2frame,
    hue='method',)
ax.legend(fontsize=12)
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# plt.show()
plt.ylabel('Eval')
# plt.xlabel('Eval type')
# plt.xticks(rotation=35, fontsize=8)
# plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.3)
# plt.style.use('ggplot')
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_DFO.svg')
# fig.show()


# fig = plt.figure(figsize=(6, 5))
# ax = plt.subplot(111)
# sns.barplot(x="method", 
#     y="eval",
#     data=DFO2frame,
#     hue='type',)

# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()
# plt.ylabel('Eval')
# plt.xlabel('Eval type')
# # plt.xticks(rotation=35, fontsize=8)
# # plt.tight_layout()
# # plt.gcf().subplots_adjust(bottom=0.3)
# fig.savefig(save_fig+'DFO_eval2.svg')
# fig.show()



evaldict = {
    'eval': [],
    'type': [],
    'method': [],
}

eval_type = ['upper', 'bi', 'tri', 'real']
eval_alias = ['upper', 'bi', 'approx', 'tri']

for i in range(len(dataframe)):
    evaldict['eval'] += [dataframe[type][i] for type in eval_type]
    evaldict['method'] += [dataframe['method'][i]]*4
    evaldict['type'] += eval_alias

evalframe = pd.DataFrame(evaldict)

# print(evalframe)
assert len(evalframe) == 4*len(dataframe) 

# fig = plt.figure(figsize=(15, 10))
# ax = plt.subplot(111)
# sns.barplot(x="type", 
#     y="eval",
#     data=evalframe,
#     hue='method' )


# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
# # plt.show()
# # plt.yscale('log')
# plt.ylabel('Evaluation')
# plt.xlabel('Evaluation type')
# # plt.xticks(rotation=35, fontsize=8)
# # plt.tight_layout()
# # plt.gcf().subplots_adjust(bottom=0.3)
# fig.savefig(save_fig+'opt_eval_all.svg')

surr_types = [
    'surr_bi_78_40',
    'surr_bi_42_20',
    'surr_bi_10_1',
    'surr_approx_45_2',
    'surr_approx_5_31',
    'surr_approx_5_2',
    # 'surr_tri_11_3',
    # 'surr_tri_16_1',
    # 'surr_tri_11_1',
]
dataframe1 = dataframe[dataframe['method'].isin(surr_types)]
# dataframe1.sort_values('method')

plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
sns.barplot(x="method", 
    y="opt_time",
    data=dataframe1,
    order=surr_types,
    errorbar=None )
plt.yscale('log')
plt.ylabel('Online optimization time [s]')
# plt.xlabel('  ')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
# plt.style.use('ggplot')
plt.ylim(bottom=0.001)
plt.xlabel('  ')
plt.savefig(save_fig+'opt_time_surr.svg')

evalframe1 = evalframe[evalframe['method'].isin(surr_types)]

# evalframe1.sort_values('method')

# evalframe1 = evalframe[evalframe['method'] in surr_types]



fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.barplot(x="type", 
    y="eval",
    data=evalframe1,
    hue_order=surr_types,
    hue='method' )

ax.legend()
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# plt.show()

# plt.yscale('log')
plt.ylabel('Evaluation')
# plt.xlabel('Evaluation type')
# plt.xticks(rotation=35, fontsize=8)
# plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.3)
# plt.show()
# plt.style.use('ggplot')
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_surr.svg')


all = [
    'planning', 
    'DFO_bi', 
    'DFO_approx', 
    'surr_bi_42_20',
    # 'surr_approx_45_2',
    'surr_tri_16_1',
]

all_alias = [
    'DFO_tri_init',
    'DFO_tri_bi',
    'DFO_tri_approx',
    'DFO_tri_surr_bi',
    # 'DFO_tri_surr_approx',
    'DFO_tri_surr_tri',
]


dataframe2 = dataframe[dataframe['method'].isin(all)]
# dataframe2.sort_values('method')

# plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
# sns.barplot(x="method", 
#     y="opt_time",
#     data=dataframe2,
#     errorbar=None )
# plt.yscale('log')
# plt.ylabel('Online optimization time [s]')
# # plt.xlabel('  ')
# plt.xticks(rotation=35, fontsize=12)
# # plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.25)
# # plt.style.use('ggplot')
# plt.ylim(bottom=0.001)
# plt.savefig(save_fig+'opt_time_best.svg')

# dummy_df = pd.DataFrame({'initial': [dataframe['opt_time'][to_idx[m]] for m in all],
#                          'tri DFO': [dataframe['opt_time'][to_idx[m1]] + dataframe['opt_time'][to_idx[m2]]  for i, (m1, m2) in enumerate(zip(all, all_alias))],},
#                         index=all)

# fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

# dummy_df.plot(kind='bar', stacked=True, ax=ax)
 
# plt.legend(fontsize=12)
# plt.ylabel('Online optimization time [s]')
# plt.yscale('log')
# plt.xlabel('Type')
# # plt.style.use('ggplot')
# plt.xticks(rotation=35, fontsize=12)
# # plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.25)
# plt.ylim(bottom=0.001)
# fig.savefig(save_fig+'opt_time_best_.svg')


# plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
# sns.barplot(x="method", 
#     y="opt_time",
#     data=dataframe2,
#     errorbar=None )
# plt.yscale('log')
# plt.ylabel('Online optimization time [s]')
# # plt.xlabel('  ')
# plt.xticks(rotation=35, fontsize=12)
# # plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.25)
# # plt.style.use('ggplot')
# plt.ylim(bottom=0.001)
# plt.savefig(save_fig+'opt_time_best.svg')

dummy_df = pd.DataFrame({'initial': [dataframe['real'][to_idx[m]] for m in all],
                         'tri DFO': [dataframe['real'][to_idx[m2]]  for i, (m1, m2) in enumerate(zip(all, all_alias))],},
                        index=all)


fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

# dummy_df.plot(kind='bar', stacked=True, ax=ax)
dummy_df.plot(kind='bar',  ax=ax)
 
plt.legend(fontsize=12)
plt.ylabel('Tri evaluation')
# plt.yscale('log')
# plt.xlabel('Type')
# plt.style.use('ggplot')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_best_tri.svg')


dummy_df = pd.DataFrame({'initial': [dataframe['tri'][to_idx[m]] for m in all],
                         'tri DFO': [dataframe['tri'][to_idx[m2]]  for i, (m1, m2) in enumerate(zip(all, all_alias))],},
                        index=all)

fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

dummy_df.plot(kind='bar', ax=ax)
 
plt.legend(fontsize=12)
plt.ylabel('Approx tri evaluation')
# plt.yscale('log')
# plt.xlabel('Type')
# plt.style.use('ggplot')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_best_approx.svg')

dummy_df = pd.DataFrame({'initial': [dataframe['bi'][to_idx[m]] for m in all],
                         'tri DFO': [dataframe['bi'][to_idx[m2]]  for i, (m1, m2) in enumerate(zip(all, all_alias))],},
                        index=all)

fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

dummy_df.plot(kind='bar', ax=ax)
 
plt.legend(fontsize=12)
plt.ylabel('Bi evaluation')
# plt.yscale('log')
# plt.xlabel('Type')
# plt.style.use('ggplot')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_best_bi.svg')

plt.figure(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))
sns.barplot(x="method", 
    y="opt_time",
    data=dataframe2,
    errorbar=None )
plt.yscale('log')
plt.ylabel('Online optimization time [s]')
plt.xlabel('  ')
plt.xticks(rotation=35, fontsize=12)
# plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.25)
# plt.style.use('ggplot')
plt.ylim(bottom=0.001)
plt.xlabel('  ')
plt.savefig(save_fig+'opt_time_best.svg')

dummy_df = pd.DataFrame({'initial': [dataframe['opt_time'][to_idx[m]] for m in all],
                         'tri DFO': [dataframe['opt_time'][to_idx[m1]] + dataframe['opt_time'][to_idx[m2]]  for i, (m1, m2) in enumerate(zip(all, all_alias))],},
                        index=all)

fig, ax = plt.subplots(figsize=(8, 5)) # fig = plt.figure(figsize=(6, 5))

dummy_df.plot(kind='bar', stacked=True, ax=ax)
 
plt.legend(fontsize=12)
plt.ylabel('Online optimization time [s]')
plt.yscale('log')
# plt.xlabel('Type')
# plt.style.use('ggplot')
# plt.ylim(bottom=0.001)
plt.xticks(rotation=35, fontsize=12)
plt.gcf().subplots_adjust(bottom=0.25)
plt.ylim(bottom=0.001)
plt.xlabel('  ')
fig.savefig(save_fig+'opt_time_best_.svg')


all = [
    'planning', 
    'DFO_bi', 
    'DFO_approx', 
    'surr_bi_42_20',
    'surr_approx_45_2',
    'surr_tri_16_1',
    'DFO_tri_init',
    'DFO_tri_bi',
    'DFO_tri_approx',
    'DFO_tri_surr',
    'DFO_tri_surr2',
    'DFO_tri_surr3',
]


evalframe2 = evalframe[evalframe['method'].isin(all)]
evalframe2.sort_values('method')

# evalframe1 = evalframe[evalframe['method'] in surr_types]


fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.barplot(x="type", 
    y="eval",
    data=evalframe2,
    hue='method' )
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
# # Put a legend to the right of the current axis
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend()
# plt.show()

# plt.yscale('log')
plt.ylabel('Evaluation')
# plt.xlabel('Evaluation type')
# plt.xticks(rotation=35, fontsize=8)
# plt.tight_layout()
# plt.gcf().subplots_adjust(bottom=0.3)
# plt.show()
# plt.style.use('ggplot')
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_best.svg')

methods = [
        'hierarch_11_1',
        'hierarch_11_2',
        'hierarch_11_3',
        'hierarch_12_2',
        'hierarch_12_3',
        'hierarch_16_1',
        'hierarch_22_1',
        'hierarch_24_1',
        'hierarch_26_1',
        'hierarch_42_20',
        'hierarch_63_3',
        'hierarch_79_40',
        'integrated_5_1',
        'integrated_5_2',
        # 'surr_distr/integrated_RegNN_5_23',   ### didn't work
        # 'surr_distr/integrated_RegNN_5_30',
        'integrated_5_31',
        'integrated_6_16',
        'integrated_11_1',
        'integrated_41_12',
        # 'surr_distr/integrated_RegNN_41_31',
        'integrated_45_2',
        'integrated_51_8',
        # 'surr_distr/integrated_RegNN_60_20',
        'scheduling_9_1',
        'scheduling_10_1',
        'scheduling_11_1',
        'scheduling_17_17',
        'scheduling_21_19',
        'scheduling_31_27',
        'scheduling__41_20',
        'scheduling__42_20',
        # 'surr_distr/scheduling_RegNN_50_20',
        'scheduling_56_19',
        'scheduling_78_40',
        'scheduling_79_40',
        'integrated_21_1',
        'scheduling_5_20',
        'integrated_42_1',
        'integrated_79_40',
        'integrated_37_17',
]

DFO_surr_dict = {
    'eval': [],
    'type': [],
    'method': [],
}

eval_type = ['upper', 'bi', 'tri', 'real']
eval_alias = ['upper', 'bi', 'approx', 'tri']
for i in range(len(dataframe_surr)):
    DFO_surr_dict['eval'] += [dataframe_surr[type][i] for type in eval_type]
    DFO_surr_dict['type'] += eval_alias
    if methods[i][0] == 's': DFO_surr_dict['method'] += ['bi surrogate']*4
    elif methods[i][0] == 'i': DFO_surr_dict['method'] += ['approx surrogate']*4
    else: DFO_surr_dict['method'] += ['tri surrogate']*4



fig = plt.figure(figsize=(8, 5))
ax = plt.subplot(111)
sns.boxplot(x ="type",
             y ="eval",
             hue ="method",
             hue_order = ['bi surrogate', 'approx surrogate', 'tri surrogate'],
             data = DFO_surr_dict)

ax.legend()
plt.ylabel('Evaluation')
# plt.ylim(bottom=-5, top=25)
plt.ylim(bottom=-5, top=40)
# plt.ylim(top=15)
# plt.xlabel('Evaluation type')
plt.xlabel('  ')
fig.savefig(save_fig+'opt_eval_surr_variation.svg')


