from hierarchy.plotting import plot_traj

Nt = 5
# methods = [
#     'centralized', 
#     'bi_Py-BOBYQA', 
#     # 'bi_Py-BOBYQA_low', 
#     # 'bi_Py-BOBYQA_distr', 
#     'tri_Py-BOBYQA', 
#     # 'tri_Py-BOBYQA_low', 
#     # 'tri_Py-BOBYQA_distr',
#     'hierarchical_Py-BOBYQA',
#     'hierarchical_bi',
#     'hierarchical_surr',
#     'hierarchical_surr2', 
#     'hierarchical_surr3',
# ]
# for m in methods:
#     plot_traj(m, Nt, SAVE=True)
methods = [
    # 'scheduling_RegNN_78_40',
    'scheduling_RegNN_42_20',
    # 'scheduling_RegNN_10_1',
    'integrated_RegNN_45_2',
    # 'integrated_RegNN_5_31',
    # 'integrated_RegNN_5_2',
    # 'hierarch_RegNN_11_3',
    'hierarch_RegNN_16_1',
    # 'hierarch_RegNN_11_1',
]
for m in methods:
    plot_traj(m, Nt, type='surrogate', SAVE=True)





