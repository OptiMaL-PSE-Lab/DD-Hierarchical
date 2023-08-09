
import json
import pandas as pd

from pathlib import Path

dir_data = './results/Optima/optima.csv'

# methods = [
#         'surr_distr/hierarch_RegNN_11_1',
#         'surr_distr/hierarch_RegNN_11_2',
#         'surr_distr/hierarch_RegNN_11_3',
#         'surr_distr/hierarch_RegNN_12_2',
#         'surr_distr/hierarch_RegNN_12_3',
#         'surr_distr/hierarch_RegNN_16_1',
#         'surr_distr/hierarch_RegNN_22_1',
#         'surr_distr/hierarch_RegNN_24_1',
#         'surr_distr/hierarch_RegNN_26_1',
#         'surr_distr/hierarch_RegNN_42_20',
#         'surr_distr/hierarch_RegNN_63_3',
#         'surr_distr/hierarch_RegNN_79_40',
#         'surr_distr/integrated_RegNN_5_1',
#         'surr_distr/integrated_RegNN_5_2',
#         'surr_distr/integrated_RegNN_5_31',
#         'surr_distr/integrated_RegNN_6_16',
#         'surr_distr/integrated_RegNN_11_1',
#         'surr_distr/integrated_RegNN_41_12',
#         'surr_distr/integrated_RegNN_45_2',
#         'surr_distr/integrated_RegNN_51_8',
#         'surr_distr/scheduling_RegNN_9_1',
#         'surr_distr/scheduling_RegNN_10_1',
#         'surr_distr/scheduling_RegNN_11_1',
#         'surr_distr/scheduling_RegNN_17_17',
#         'surr_distr/scheduling_RegNN_21_19',
#         'surr_distr/scheduling_RegNN_31_27',
#         'surr_distr/scheduling_RegNN_41_20',
#         'surr_distr/scheduling_RegNN_42_20',
#         'surr_distr/scheduling_RegNN_56_19',
#         'surr_distr/scheduling_RegNN_78_40',
#         'surr_distr/scheduling_RegNN_79_40',
#         'surr_distr/integrated_RegNN_21_1',
#         'surr_distr/scheduling_RegNN_5_20',
#         'surr_distr/integrated_RegNN_42_1',
#         'surr_distr/integrated_RegNN_79_40',
#         'surr_distr/integrated_RegNN_37_17',
# ]

# alias = [
#         'hierarch_11_1',
#         'hierarch_11_2',
#         'hierarch_11_3',
#         'hierarch_12_2',
#         'hierarch_12_3',
#         'hierarch_16_1',
#         'hierarch_22_1',
#         'hierarch_24_1',
#         'hierarch_26_1',
#         'hierarch_42_20',
#         'hierarch_63_3',
#         'hierarch_79_40',
#         'integrated_5_1',
#         'integrated_5_2',
#         'integrated_5_31',
#         'integrated_6_16',
#         'integrated_11_1',
#         'integrated_41_12',
#         'integrated_45_2',
#         'integrated_51_8',
#         'scheduling_9_1',
#         'scheduling_10_1',
#         'scheduling_11_1',
#         'scheduling_17_17',
#         'scheduling_21_19',
#         'scheduling_31_27',
#         'scheduling__41_20',
#         'scheduling__42_20',
#         'scheduling_56_19',
#         'scheduling_78_40',
#         'scheduling_79_40',
#         'integrated_21_1',
#         'scheduling_5_20',
#         'integrated_42_1',
#         'integrated_79_40',
#         'integrated_37_17',
# ]

methods = [
    'centralized', 
    'bi_Py-BOBYQA', 
    'bi_Py-BOBYQA_low', 
    'bi_Py-BOBYQA_distr', 
    'tri_Py-BOBYQA', 
    'tri_Py-BOBYQA_low', 
    'tri_Py-BOBYQA_distr',
    'hierarchical_Py-BOBYQA',
    'hierarchical_init',
    'hierarchical_bi',
    'hierarchical_surr',
    'hierarchical_surr2',
    'hierarchical_surr3',
    'integrated_RegNN_45_2',
    'integrated_RegNN_5_31',
    'integrated_RegNN_5_2',
    'scheduling_RegNN_78_40',
    'scheduling_RegNN_42_20',
    'scheduling_RegNN_10_1',
    'hierarch_RegNN_11_3',
    'hierarch_RegNN_16_1',
    'hierarch_RegNN_11_1',
]

alias = [
    'planning', 
    'DFO_bi', 
    'DFO_bi_low', 
    'DFO_bi_distr', 
    'DFO_approx', 
    'DFO_approx_low', 
    'DFO_approx_distr',
    'DFO_tri_approx',
    'DFO_tri_init',
    'DFO_tri_bi',
    'DFO_tri_surr_approx',
    'DFO_tri_surr_bi',
    'DFO_tri_surr_tri',
    'surr_approx_45_2',
    'surr_approx_5_31',
    'surr_approx_5_2',
    'surr_bi_78_40',
    'surr_bi_42_20',
    'surr_bi_10_1',
    'surr_tri_11_3',
    'surr_tri_16_1',
    'surr_tri_11_1',
]


data = {
    'opt_time': [],
    'method': [],
    'upper': [],
    'bi': [],
    'tri': [],
    'real': [],
}

for i,m in enumerate(methods):
    dir = f"./results/Optima/{m}.json"
    if Path(dir).is_file():
        
        try: 
            with open(dir, 'r+') as file: 
                file_data = json.load(file)
        except: 
            print(f"Problem loading model {m}")
        data['method'] += [alias[i]]
        data['real'] += [file_data['real']['obj']]

        if 'time' in file_data: 
            data['opt_time'] += [file_data['time']]
            data['tri'] += [file_data['tri']['obj']]
            data['bi'] += [file_data['bi']['obj']]
            data['upper'] += [file_data['upper']['obj']]
        elif 'opt_time_l' in file_data: 
            data['opt_time'] += [file_data['opt_time_l']]
            data['tri'] += [file_data['large']['tri']['obj']]
            data['bi'] += [file_data['large']['bi']['obj']]
            data['upper'] += [file_data['large']['upper']['obj']]
        elif 'opt_time_s' in file_data:  
            data['opt_time'] += [file_data['opt_time_s']]
            data['tri'] += [file_data['hierarchy']['tri']['obj']]
            data['bi'] += [file_data['hierarchy']['bi']['obj']]
            data['upper'] += [file_data['hierarchy']['upper']['obj']]
    else:
        print(m, ' could not be read')


dataframe = pd.DataFrame(data=data)
dataframe.to_csv(dir_data)







