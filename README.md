# DD-Hierarchical
Exploring derivative-free optimization (DFO) and optimality surrogates to solve tri-level formulations of a hierarchical planning-scheduling-control formulation.
This repository supports the 'Hierarchical Planning-Scheduling-Control - Derivative-free optimization and optimality surrogates' work.

## Installation and dependencies

 Using `conda env create --file environment.yml` will recreate an environment with the required dependencies.

## Using the repository

The repository is divided into the data, hierarchy, results, and scripts folders.

- data: Contains the planning, and scheduling optimization model parameters. Also includes optimal control, scheduling, and scheduling-control data sampled by running the corresponding optimization problems for a range of planning and scheduling setpoints used for surrogate training
- hierarchy: Contains planning, scheduling, and control optimization Pyomo formulations with any integrated variations, any DFO wrappers, and plotting utilities
- results: Contains the DFO and optimality surrogate results, trained optimality surrogates, and result plots
- scripts: Contains all the scripts from scheduling and control optimization sampling, to surrogate training, tri-level optimization (with the .pbs scripts to give an idea of the required resources) and plotting

- ## Worflows

- Sampling the optimization problems to create optimality surrogates: `sampling.py` (control), `sampling_Sch.py` (scheduling), `sampling_integrated.py` (scheduling with control surrogate), `sampling_hierarch.py` (sequential scheduling and control)
- Training optimality surrogates on a range of planning/scheduling setpoints on the sampling_*.py samples stored in data: `training.py`, `training_Sch.py`, `training_Sch_NN.py`, `training_Sch_hierarch.py`
- Sampling the hierarchical tri-level evaluation for a given planning-level solution: `sample_real___.py`
- Hyperparameter tuning of and tri-level optimization with optimality surrogates: `surrogate_pareto.py` (scheduling-only), `surrogate_integr_pareto1.py`, `surrogate_integr_pareto2.py` (scheduling-approximate control), `surrogate_hierarch_pareto.py` (sequential scheduling-control)
- DFO: on the planning, planning-scheduling and planning-scheduling-approximate control problems on a simplified, simplified but parallelized, and the full parallelized case study: `DFO_ext_lowdim.py`, `DFO_ext_lowdim_parallel.py`, `DFO_ext_all.py`; on the hierarchical tri-level planning-scheduling-control from the planning-only, planning-scheduling, planning-scheduling-approximate control DFO solutions and the surrogate solutions: `DFO_real_init.py`, `DFO_real_bi.py`, `DFO_real_.py`, `DFO_real_surr.py`
- Plotting: `save_optima_data.py` saves the results in a dataformatused in `plot_optima.py` to create all plots to support the work
