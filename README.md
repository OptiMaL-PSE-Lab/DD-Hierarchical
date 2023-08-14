# DD-Hierarchical
Exploring derivative-free optimization (DFO) and optimality surrogates to solve tri-level formulations of a hierarchical planning-scheduling-control formulation.
This repository supports the 'Hierarchical Planning-Scheduling-Control - Derivative-free optimization and optimality surrogates' work.

## Installation and dependencies

 Using '''conda env create --file environment.yml''' will recreate an environment with the required dependencies.

## Using the repository

The repository is divided into the data, hierarchy, results, and scripts folders.

- data: Contains the planning, and scheduling optimization model parameters. Also includes optimal control, scheduling, and scheduling-control data sampled by running the corresponding optimization problems for a range of planning and scheduling setpoints used for surrogate training
- hierarchy: Contains planning, scheduling, and control optimization Pyomo formulations with any integrated variations, any DFO wrappers, and plotting utilities
- results: Contains the DFO and optimality surrogate results, trained optimality surrogates, and result plots
- scripts: Contains all the scripts from scheduling and control optimization sampling, to surrogate training, tri-level optimization (with the .pbs scripts to give an idea of the required resources) and plotting

- ## Worflows

- Sampling the optimization problems to create optimality surrogates: sampling.py (control), sampling_Sch.py (scheduling), sampling_integrated.py (scheduling with control surrogate), sampling_hierarch.py (sequential scheduling and control)
- Training optimality surrogates on a range of planning/scheduling setpoints on the sampling_*.py samples stored in data: training.py, training_Sch.py, training_Sch_NN.py, training_Sch_hierarch.py
- 
