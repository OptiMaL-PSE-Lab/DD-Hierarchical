import numpy as np

products = ["PA", "PB", "PC", "PD", "TEE", "TGE"]
secondary_sites = ["Asia", "America"]
resources = ["Granulates", "Compress", "Coat", "QC", "Packing"]
# intermediates = ['TA', 'TB', 'I1', 'I2', 'I3', 'I4']
T_set = np.arange(1, 13).tolist()


# F = {("PA", t): (t - 1) * 4000 / 100 + 5000 for t in T_set}
# F_ = {("PC", t): (t - 1) * 1e4 / 100 + 10000 for t in T_set}
# F__ = {("TEE", t): (t - 1) * 1e5 / 100 + 400000 for t in T_set}
F = {("PA", t): 5000*(1+.1*int(t>=4) - .2*int(t>=7) + .3*int(t>=9) - .4*int(t>=11)) for t in T_set}
F_ = {("PC", t): 10000*(1+.1*int(t>=4) - .2*int(t>=7) + .3*int(t>=9) - .4*int(t>=11)) for t in T_set}
F__ = {("TEE", t): 300000*(1+.1*int(t>=4) - .2*int(t>=7) + .3*int(t>=9) - .4*int(t>=11)) for t in T_set}
F_TGE = {("TGE", t): 300000*(1-.1*int(t>=4) + .2*int(t>=7) - .3*int(t>=9) + .4*int(t>=11)) for t in T_set}
F_PB = {("PB", t): 5000*(1-.1*int(t>=4) + .2*int(t>=7) - .3*int(t>=9) + .4*int(t>=11)) for t in T_set}
F_PD = {("PD", t): 10000*(1-.1*int(t>=4) + .2*int(t>=7) - .3*int(t>=9) + .4*int(t>=11)) for t in T_set}
F.update(F_) ; F.update(F__) ; F.update(F_TGE)
F.update(F_PB) ; F.update(F_PD)

data = {
    None: {
        "P": {None: products},
        "L": {None: secondary_sites},
        "R": {None: resources},
        "T": {None: T_set},
        # Initial storage
        "S0": {
            "PA": 15e3, "PB": 15e3,
            "PC": 6e4, "PD": 6e4,
            "TEE": 2e6, "TGE": 2e6}, 
        "SAIP0": {None: 3000},
        "SI0": {None: 3000},
        "SAIS0": {"Asia": 480, "America": 360},
        #Safety storage
        "IAIPstar0": {None: 3000},
        "IIstar0": {None: 3000},
        "Istar0": {
            "PA": 15e3, "PB": 15e3, "PC": 6e4, "PD": 6e4, 
            "TEE": 2e6, "TGE": 2e6,
            },
        "IAISstar0": {"Asia": 400, "America": 300},
        #Costs
        "CT": {"Asia": 15, "America": 10}, #Transport
        "CS": {"PA": 0.1, "PC": 0.15, "TEE": 0.1,
                "PB": 0.09, "PD": 0.16, "TGE": 0.1,}, # Storage
        "CS_SAIS": {"Asia": 0.02, "America": 0.03},
        # 'CS_SAIS': {'Asia': 0, 'America': 0},
        "CS_AIP": {None: 0.02},
        "CS_I": {None: 0.01},
        # 'CS_AIP': {None: 0}, 'CS_I': {None: 0},
        "RM_Cost": {None: 0.1}, #Raw material cost
        "AI_Cost": {None: 0.5}, # AI cost (should not matter for the whole company)
        # 'RM_Cost': {None: 0}, 'AI_Cost': {None: 0},
        "Price": {"PA": 10, "PB": 10, "PC": 10, "PD": 10,
                  "TEE": 1, "TGE": 1,},
        "CP": {"PA": 1.2, "PC": 1.2, "TEE": 0.06,
               "PB": 1.19, "PD": 1.21, "TGE": 0.06,}, # Production
        "CP_I": {None: 0.05e-1},
        "CP_AI": {None: 0.2e-1},
        "SP_AI": {None: 2500}, # Selling price
        # 'SP_AI': {None: 0},
        "LT": {"Asia": 4, "America": 3}, # lead time
        "Q": {"PA": 20e-6 * 12 / 0.02, "PC": 6e-4 / 0.02, "TEE": 3e-5,
              "PB": 21e-6 * 12 / 0.02, "PD": 5.9e-4 / 0.02, "TGE": 3.1e-5,}, # AI to material conversion
        # No FPAI(T), FPTA(T), FPTB(T), FPI4(T), FPI3(T), FPI2(T), FPI1(T), FTP(T,K)
        "A": {  # Resource availability
            ("Asia", "Granulates"): 120*2,
            ("Asia", "Compress"): 480*2,
            ("Asia", "Coat"): 480*2,
            ("Asia", "QC"): 1800*2,
            ("Asia", "Packing"): 320*2,
            ("America", "Granulates"): 120*2,
            ("America", "Compress"): 120*2,
            ("America", "Coat"): 120*2,
            ("America", "QC"): 720*2,
            ("America", "Packing"): 160*2,
        },
        "U": { # Resource time utilised per product
            ("Asia", "Granulates"): 133e-3,
            ("Asia", "Compress"): 350e-3,
            ("Asia", "Coat"): 333e-3,
            ("Asia", "QC"): 2,
            ("Asia", "Packing"): 267e-3,
            ("America", "Granulates"): 133e-3,
            ("America", "Compress"): 350e-3,
            ("America", "Coat"): 333e-3,
            ("America", "QC"): 2,
            ("America", "Packing"): 267e-3,
        },
        "CCH": {
            ("PA", "PA"): 0,
            ("PA", "PB"): 3e4,
            ("PA", "PC"): 1e8,
            ("PA", "PD"): 1e8,
            ("PA", "TEE"): 2.9e4,
            ("PA", "TGE"): 2.9e4,
            ("PB", "PA"): 3e4,
            ("PB", "PB"): 0,
            ("PB", "PC"): 1e8,
            ("PB", "PD"): 1e8,
            ("PB", "TEE"):2.9e4,
            ("PB", "TGE"):2.9e4,
            ("PC", "PA"): 1e8,
            ("PC", "PB"): 1e8,
            ("PC", "PC"): 0,
            ("PC", "PD"): 1e4,
            ("PC", "TEE"):1e8,
            ("PC", "TGE"):1e8,
            ("PD", "PA"): 1e8,
            ("PD", "PB"): 1e8,
            ("PD", "PC"): 1e4,
            ("PD", "PD"): 0,
            ("PD", "TEE"): 1e8,
            ("PD", "TGE"): 1e8,
            ("TEE", "PA"): 3.1e4,
            ("TEE", "PB"): 3.1e4,
            ("TEE", "PC"): 1e8,
            ("TEE", "PD"): 1e8,
            ("TEE", "TEE"): 0,
            ("TEE", "TGE"): 3e4,
            ("TGE", "PA"): 3.1e4,
            ("TGE", "PB"): 3.1e4,
            ("TGE", "PC"): 1e8,
            ("TGE", "PD"): 1e8,
            ("TGE", "TEE"): 3e4,
            ("TGE", "TGE"): 0,
        },
        "X": { # matches between final products and secondary sites
            ("Asia", "PA"): 1,
            ("Asia", "PC"): 0,
            ("Asia", "TEE"): 1,
            ("Asia", "PB"): 1,
            ("Asia", "PD"): 0,
            ("Asia", "TGE"): 1,
            ("America", "PA"): 0,
            ("America", "PC"): 1,
            ("America", "TEE"): 0,
            ("America", "PB"): 0,
            ("America", "PD"): 1,
            ("America", "TGE"): 0,
        },
        "F": F, # Demand forecast
    }
}

np.random.seed(0)
I_list = ['TI1', 'TI2', 'TEE1', 'TEE2', 'TGE1', 'TGE2', 'PA1', 'PA2', 'PB1', 'PB2']
S_list = ['AI', 'TI', 'TEE', 'TGE', 'PA', 'PB']
gridpoints = 7 # 20 initially

rho_dict = {(i,s): 0 for s in S_list for i in I_list}
rho_dict[('TI1','TI')] = 1e3 ; rho_dict[('TI2','TI')] = 1e3
rho_dict[('TI1','AI')] = 1.01e-2 ; rho_dict[('TI2','AI')] = 1e-2
rho_dict[('TEE1','TEE')] = 1e3 ; rho_dict[('TEE2','TEE')] = 1e3
rho_dict[('TEE1','AI')] = 3e-2 ; rho_dict[('TEE2','AI')] = 3.1e-2
rho_dict[('TGE1','TGE')] = 1e3 ; rho_dict[('TGE2','TGE')] = 1e3
rho_dict[('TGE1','AI')] = 2.9e-2 ; rho_dict[('TGE2','AI')] = 3e-2
rho_dict[('PA1','PA')] = 1e3 ; rho_dict[('PA2','PA')] = 1e3
rho_dict[('PA1','TI')] = 20e-6/1.01e-5 * 12 / 0.02 *1e3
rho_dict[('PA2','TI')] = 20.2e-6/1e-5 * 12 / 0.02  *1e3
rho_dict[('PB1','PB')] = 1e3 ; rho_dict[('PB2','PB')] = 1e3
rho_dict[('PB1','TI')] = 20.9e-6/1.01e-5 * 12 / 0.02 *1e3
rho_dict[('PB2','TI')] = 21e-6/1e-5 * 12 / 0.02      *1e3

scheduling_data = {
    None: {
        'N': {None: np.arange(gridpoints).tolist()},
        'R': {None: ['U1', 'U2']},
        'states': {None: S_list},
        'I': {None: I_list},
        'tasks': {None:
            [('U1','TI1'), ('U1','TEE1'), ('U1','TGE1'), ('U1','PA1'), ('U1','PB1'),
            ('U2','TI2'), ('U2','TEE2'), ('U2','TGE2'), ('U2','PA2'), ('U2','PB2'),]
         },
        'N_last': {None: gridpoints-1},
        "S0": {
            "PA": 15e3*0, "PB": 15e3*0,
            "TI": 0, "AI": 480,
            "TEE": 2e6*0, "TGE": 2e6*0
            },
        'alpha': {i: 2*(1+0.1*np.random.normal()) for i in I_list},
        'beta': {
            'TI1':  40/1e6*1e3, 'TI2':  40/1e6*1e3,
            'TEE1': 40/2e5*1e3, 'TEE2': 40/2e5*1e3,
            'TGE1': 40/2e5*1e3, 'TGE2': 40/2e5*1e3,
            'PA1':  60/3e3*1e3, 'PA2':  60/3e3*1e3,
            'PB1':  60/3e3*1e3, 'PB2':  60/3e3*1e3,
        },

        'H': {None: 730},
        ## Relax constraints to max
        'Bmin': {
            'TI1':  1e-3*40000,
            'TI2':  1e-3*40000,
            'TEE1': 1e-3*20000,
            'TEE2': 1e-3*20000,
            'TGE1': 1e-3*20000,
            'TGE2': 1e-3*20000,
            'PA1':  1e-3*300,
            'PA2':  1e-3*300,
            'PB1':  1e-3*300,
            'PB2':  1e-3*300,
        },
        'Bmax': {
            'TI1': 1e-3*10e6*(1+0.1*np.random.normal()), # change to 1e6
            'TI2': 1e-3*10e6*(1+0.1*np.random.normal()), # change to 1e6
            'TEE1':1e-3* 200000*(1+0.1*np.random.normal()),
            'TEE2':1e-3* 200000*(1+0.1*np.random.normal()),
            'TGE1':1e-3* 200000*(1+0.1*np.random.normal()),
            'TGE2':1e-3* 200000*(1+0.1*np.random.normal()),
            'PA1': 1e-3*3000*(1+0.1*np.random.normal()),
            'PA2': 1e-3*3000*(1+0.1*np.random.normal()),
            'PB1': 1e-3*3000*(1+0.1*np.random.normal()),
            'PB2': 1e-3*3000*(1+0.1*np.random.normal()),
            },
        ##
        'rho': rho_dict,
        'S_in': {None: 
            [('TI1','AI'), ('TI2','AI'),
            ('TEE1','AI'), ('TEE2','AI'),
            ('TGE1','AI'), ('TGE2','AI'),
            ('PA1','TI'), ('PA2', 'TI'),
            ('PB1','TI'), ('PB2', 'TI')],
        },
        'S_out': {None:
            [('TI1','TI'), ('TI2','TI'),
            ('TEE1','TEE'), ('TEE2','TEE'),
            ('TGE1','TGE'), ('TGE2','TGE'),
            ('PA1','PA'), ('PA2','PA'),
            ('PB1','PB'), ('PB2','PB')],
        },
        'in_s': {None:
            [('AI','TI1'), ('AI','TI2'), ('AI','TEE1'), ('AI','TEE2'), ('AI','TGE1'), ('AI','TGE2'), 
            ('TI','PA1'), ('TI','PA2'), ('TI','PB1'), ('TI','PB2')]
            # 'TEE': [],'TGE': [], 'PA': [], 'PB': [],]
        },
        'out_s': {None:
            # 'AI': [], 
            [('TI','TI1'), ('TI','TI2'),
            ('TEE','TEE1'), ('TEE','TEE2'), ('TGE','TGE1'), ('TGE','TGE2'),
            ('PA','PA1'), ('PA','PA2'), ('PB','PB1'), ('PB','PB2')],
        },
        'Prod': {s: 0 for s in S_list},
        'proc_time': {
                ('TI1','TI1'): 0, ('TI1','TEE1'): 3.9, ('TI1','TGE1'): 4, ('TI1','PA1'): 3.5, ('TI1','PB1'): 3.4,
                ('TI1','TI2'): 1e5, ('TI1','TEE2'): 1e5, ('TI1','TGE2'): 1e5, ('TI1','PA2'): 1e5, ('TI1','PB2'): 1e5,
                ('TEE1','TI1'): 4.1, ('TEE1','TEE1'): 0, ('TEE1','TGE1'): 4.2, ('TEE1','PA1'): 3.8, ('TEE1','PB1'): 3.7,
                ('TEE1','TI2'): 1e5, ('TEE1','TEE2'): 1e5, ('TEE1','TGE2'): 1e5, ('TEE1','PA2'): 1e5, ('TEE1','PB2'): 1e5,
                ('TGE1','TI1'): 4.1, ('TGE1','TEE1'): 4.2, ('TGE1','TGE1'): 0, ('TGE1','PA1'): 3.8, ('TGE1','PB1'): 3.9,
                ('TGE1','TI2'): 1e5, ('TGE1','TEE2'): 1e5, ('TGE1','TGE2'): 1e5, ('TGE1','PA2'): 1e5, ('TGE1','PB2'): 1e5,
                ('PA1','TI1'): 3.1+2, ('PA1','TEE1'): 3.2+2, ('PA1','TGE1'): 5, ('PA1','PA1'): 0, ('PA1','PB1'): 4.1,
                ('PA1','TI2'): 1e5, ('PA1','TEE2'): 1e5, ('PA1','TGE2'): 1e5, ('PA1','PA2'): 1e5, ('PA1','PB2'): 1e5,
                ('PB1','TI1'): 3.1+2.1, ('PB1','TEE1'): 3+2, ('PB1','TGE1'): 5.1, ('PB1','PA1'): 3.9, ('PB1','PB1'): 0,
                ('PB1','TI2'): 1e5, ('PB1','TEE2'): 1e5, ('PB1','TGE2'): 1e5, ('PB1','PA2'): 1e5, ('PB1','PB2'): 1e5,

                ('TI2','TI2'): 0, ('TI2','TEE2'): 4, ('TI2','TGE2'): 4.1, ('TI2','PA2'): 3.4, ('TI2','PB2'): 3.5,
                ('TI2','TI1'): 1e5, ('TI2','TEE1'): 1e5, ('TI2','TGE1'): 1e5, ('TI2','PA1'): 1e5, ('TI2','PB1'): 1e5,
                ('TEE2', 'TI2'): 4.2, ('TEE2','TEE2'): 0, ('TEE2','TGE2'): 4.2, ('TEE2','PA2'): 3.8, ('TEE2','PB2'): 3.8,
                ('TEE2','TI1'): 1e5, ('TEE2','TEE1'): 1e5, ('TEE2','TGE1'): 1e5, ('TEE2','PA1'): 1e5, ('TEE2','PB1'): 1e5,
                ('TGE2','TI2'): 4, ('TGE2','TEE2'): 4.1, ('TGE2','TGE2'): 0, ('TGE2','PA2'): 3.8, ('TGE2','PB2'): 3.7,
                ('TGE2','TI1'): 1e5, ('TGE2','TEE1'): 1e5, ('TGE2','TGE1'): 1e5, ('TGE2','PA1'): 1e5, ('TGE2','PB1'): 1e5,
                ('PA2','TI2'): 3.1+2.1, ('PA2','TEE2'): 3.2+2.1, ('PA2','TGE2'): 5, ('PA2','PA2'): 0, ('PA2','PB2'): 4,
                ('PA2','TI1'): 1e5, ('PA2','TEE1'): 1e5, ('PA2','TGE1'): 1e5, ('PA2','PA1'): 1e5, ('PA2','PB1'): 1e5,
                ('PB2','TI2'): 3.1+1.9, ('PB2','TEE2'): 3+1.9, ('PB2','TGE2'): 5.2, ('PB2','PA2'): 3.9, ('PB2','PB2'): 0,
                ('PB2','TI1'): 1e5, ('PB2','TEE1'): 1e5, ('PB2','TGE1'): 1e5, ('PB2','PA1'): 1e5, ('PB2','PB1'): 1e5,
        },
        'kappa': {
            ('TI1','TI1'): 0, ('TI1','TEE1'): 2.9e4, ('TI1','TGE1'): 2.8e4, ('TI1','PA1'): 2.5e4, ('TI1','PB1'): 2.6e4,
            ('TI1','TI2'): 1e8, ('TI1','TEE2'): 1e8, ('TI1','TGE2'): 1e8, ('TI1','PA2'): 1e8, ('TI1','PB2'): 1e8,
            ('TEE1','TI1'): 3.1e4, ('TEE1','TEE1'): 0, ('TEE1','TGE1'): 3.2e4, ('TEE1','PA1'): 2.6e4, ('TEE1','PB1'): 2.7e4,
            ('TEE1','TI2'): 1e8, ('TEE1','TEE2'): 1e8, ('TEE1','TGE2'): 1e8, ('TEE1','PA2'): 1e8, ('TEE1','PB2'): 1e8,
            ('TGE1','TI1'): 3.2e4, ('TGE1','TEE1'): 3.1e4, ('TGE1','TGE1'): 0, ('TGE1','PA1'): 2.9e4, ('TGE1','PB1'): 2.8e4,
            ('TGE1','TI2'): 1e8, ('TGE1','TEE2'): 1e8, ('TGE1','TGE2'): 1e8, ('TGE1','PA2'): 1e8, ('TGE1','PB2'): 1e8,
            ('PA1','TI1'): 3.2e4+1e4, ('PA1','TEE1'): 3.1e4+1e4, ('PA1','TGE1'): 4e4, ('PA1','PA1'): 0, ('PA1','PB1'): 3.1e4,
            ('PA1','TI2'): 1e8, ('PA1','TEE2'): 1e8, ('PA1','TGE2'): 1e8, ('PA1','PA2'): 1e8, ('PA1','PB2'): 1e8,
            ('PB1','TI1'): 3.1e4+1.1e4, ('PB1','TEE1'): 3e4+1e4, ('PB1','TGE1'): 4.1e4, ('PB1','PA1'): 2.9e4, ('PB1','PB1'): 0,
            ('PB1','TI2'): 1e8, ('PB1','TEE2'): 1e8, ('PB1','TGE2'): 1e8, ('PB1','PA2'): 1e8, ('PB1','PB2'): 1e8,

            ('TI2','TI2'): 0, ('TI2','TEE2'): 3.1e4, ('TI2','TGE2'): 3e4, ('TI2','PA2'): 2.5e4, ('TI2','PB2'): 2.4e4,
            ('TI2','TI1'): 1e8, ('TI2','TEE1'): 1e8, ('TI2','TGE1'): 1e8, ('TI2','PA1'): 1e8, ('TI2','PB1'): 1e8,
            ('TEE2', 'TI2'): 3.2e4, ('TEE2','TEE2'): 0, ('TEE2','TGE2'): 3.2e4, ('TEE2','PA2'): 2.8e4, ('TEE2','PB2'): 2.8e4,
            ('TEE2','TI1'): 1e8, ('TEE2','TEE1'): 1e8, ('TEE2','TGE1'): 1e8, ('TEE2','PA1'): 1e8, ('TEE2','PB1'): 1e8,
            ('TGE2','TI2'): 3.1e4, ('TGE2','TEE2'): 3e4, ('TGE2','TGE2'): 0, ('TGE2','PA2'): 2.7e4, ('TGE2','PB2'): 2.8e4,
            ('TGE2','TI1'): 1e8, ('TGE2','TEE1'): 1e8, ('TGE2','TGE1'): 1e8, ('TGE2','PA1'): 1e8, ('TGE2','PB1'): 1e8,
            ('PA2','TI2'): 3.1e4+1.1e4, ('PA2','TEE2'): 3.2e4+1.1e4, ('PA2','TGE2'): 4.5e4, ('PA2','PA2'): 0, ('PA2','PB2'): 3e4,
            ('PA2','TI1'): 1e8, ('PA2','TEE1'): 1e8, ('PA2','TGE1'): 1e8, ('PA2','PA1'): 1e8, ('PA2','PB1'): 1e8,
            ('PB2','TI2'): 3.1e4+.9e4, ('PB2','TEE2'): 3e4+.9e4, ('PB2','TGE2'): 4.2e4, ('PB2','PA2'): 2.9e4, ('PB2','PB2'): 0,
            ('PB2','TI1'): 1e8, ('PB2','TEE1'): 1e8, ('PB2','TGE1'): 1e8, ('PB2','PA1'): 1e8, ('PB2','PB1'): 1e8,
        },
        "CS": {"PA": 0.1, "AI": 0.01, "TEE": 0.1,
                "PB": 0.09, "TI": 0.02, "TGE": 0.1,}, # Storage
    }
}

