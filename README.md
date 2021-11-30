# TermPaper_ineq
Code for the project "Does this distribution make my tail look fat: Estimation of the Gini Coefficient for Fat Tailed Data.
The project is written for the seminar "Inequality and Macroeconomics" at Copenhagen University, Department of Economics.

## Overview of the notebooks
- 00 - ParetoExponential.ipynb illustrates implications of fat tails for Section 2
- 01 - EstimatorMC.ipynb performs a Monte Carlo simulation for the nonparametric and semiparametric estimators for Section 3
- 02 - ModelMC performs.ipynb rejection sampling from the stationary wealth distribution of the HA model, and a Monte Carlo simulation for the estimators for Section 4
- functions.py contains a list of functions used in the notebooks
- model.m is a script to run the HA model and generate the stationary distribution. It is based on the original script found at https://benjaminmoll.com/wp-content/uploads/2020/06/fat_tail_partialeq.m from the personal site of Benjamin Moll
- a_var.csv and ga_var.csv contains the wealth distribution computed in model.m. These are imported in python in 02 - ModelMC.ipynb.
