
import json
import pandas as pd

from pathlib import Path

dir_data = './results/Models/dataframe.csv'

nodes1 = [10, 20, 30, 40, 50, 80]
nodes2 = [1, 5, 10, 20, 30, 40]

data = {
    'time_s': [],
    'time_l': [],
    'test_score': [],
    'nodes': [],
    'type': [],
    'upper_s': [],
    'upper_l': [],
    'bi_s': [],
    'bi_l': [],
    'tri_s': [],
    'tri_l': [],
    'real': [],
}


for n1 in range(nodes1[-1]):
    for n2 in range(nodes2[-1]):
        dir = f"./results/Models/scheduling_RegNN_{n1+1}_{n2+1}.json"
        if Path(dir).is_file():
            try:
                with open(dir, 'r+') as file:
                    file_data = json.load(file)
            except: print(f"Problem loading model with nodes {(n1, n2)}")
            data['test_score'] += [file_data['test_score']]
            data['type'] += ['schedu']
            data['nodes'] += [(n1+1, n2+1)]
            if 'opt_time_l' in file_data:
                data['time_l'] += [file_data['opt_time_l']]
                data['upper_l'] += [file_data['large']['upper']['obj']]
                data['bi_l'] += [file_data['large']['bi']['obj']]
                data['tri_l'] += [file_data['large']['tri']['obj']]
            else:
                data['time_l'] += [None]
                data['upper_l'] += [None]
                data['bi_l'] += [None]
                data['tri_l'] += [None]
            if 'opt_time_s' in file_data:
                data['time_s'] += [file_data['opt_time_s']]
                data['upper_s'] += [file_data['hierarchy']['upper']['obj']]
                data['bi_s'] += [file_data['hierarchy']['bi']['obj']]
                data['tri_s'] += [file_data['hierarchy']['tri']['obj']]
            else:
                data['time_s'] += [None]
                data['upper_s'] += [None]
                data['bi_s'] += [None]
                data['tri_s'] += [None]
            if 'real' in file_data:
                data['real'] += [file_data['real']['obj']]
            else:
                data['real'] += [None]
        dir = f"./results/Models/integrated_RegNN_{n1+1}_{n2+1}.json"
        if Path(dir).is_file():
            with open(dir, 'r+') as file:
                try:
                    file_data = json.load(file)
                except: print(n1, n2)
            data['test_score'] += [file_data['test_score']]
            data['type'] += ['integr']
            data['nodes'] += [(n1+1, n2+1)]
            if 'opt_time_l' in file_data:
                data['time_l'] += [file_data['opt_time_l']]
                data['upper_l'] += [file_data['large']['upper']['obj']]
                data['bi_l'] += [file_data['large']['bi']['obj']]
                data['tri_l'] += [file_data['large']['tri']['obj']]
            else:
                data['time_l'] += [None]
                data['upper_l'] += [None]
                data['bi_l'] += [None]
                data['tri_l'] += [None]
            if 'opt_time_s' in file_data:
                data['time_s'] += [file_data['opt_time_s']]
                data['upper_s'] += [file_data['hierarchy']['upper']['obj']]
                data['bi_s'] += [file_data['hierarchy']['bi']['obj']]
                data['tri_s'] += [file_data['hierarchy']['tri']['obj']]
            else:
                data['time_s'] += [None]
                data['upper_s'] += [None]
                data['bi_s'] += [None]
                data['tri_s'] += [None]
            if 'real' in file_data:
                data['real'] += [file_data['real']['obj']]
            else:
                data['real'] += [None]

dataframe = pd.DataFrame(data=data)
dataframe.to_csv(dir_data)







