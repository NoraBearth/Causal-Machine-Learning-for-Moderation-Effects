import argparse
import numpy as np
from tqdm import tqdm
import os
from src import data, estimators, models

# parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument("--num_simulations", type=int, default=100)
argparser.add_argument("--n", type=int, default=1000)
argparser.add_argument("--p", type=int, default=10)
argparser.add_argument("--mode", type=int, default=2)
argparser.add_argument("--n_folds", type=int, default=2)
argparser.add_argument("--model_name", type=str, default="RandomForest")
argparser.add_argument("--seed", type=int, default=256)
argparser.add_argument("--n_jobs", type=int, default=-1)
argparser.add_argument("--num_simulations_hyperparam", type=int, default=10)
argparser.add_argument("--balancing_variables", type=lambda s: list(map(int, s.split(','))), default=None)
args = argparser.parse_args()

model_args = {"random_state": args.seed}

if args.model_name == "RandomForest":
    model_args = {"n_jobs": args.n_jobs, **model_args}

# set seed for hyperparameter tuning
np.random.seed(args.seed)
# list to store optimal hyperparameters
hparams_reg_treated = []
hparams_reg_not_treated = []
hparams_propensity = []
hparams_reg_Z1 = []
hparams_reg_Z0 = []
hparams_propensity_Z = []


# add progress bar
progress_bar = tqdm(total=args.num_simulations_hyperparam, desc="Hyperparameter tuning")
for i in range(args.num_simulations_hyperparam):
    y, d, z, x = data.simulate_data(
        n=args.n, p=args.p, mode=args.mode,
        balancing_variables=args.balancing_variables, true_effect=False)

    (hparams_reg_treated_i, hparams_reg_not_treated_i, hparams_propensity_i,
     hparams_reg_Z1_i, hparams_reg_Z0_i, hparams_propensity_Z_i) = (
        models.tune_nuisances(y, d, x, z, args.balancing_variables,
                              nfolds=args.n_folds, model_name=args.model_name,
                              n_jobs_cv=args.n_jobs, **model_args,))
    hparams_reg_treated.append(hparams_reg_treated_i)
    hparams_reg_not_treated.append(hparams_reg_not_treated_i)
    hparams_propensity.append(hparams_propensity_i)
    hparams_reg_Z1.append(hparams_reg_Z1_i)
    hparams_reg_Z0.append(hparams_reg_Z0_i)
    hparams_propensity_Z.append(hparams_propensity_Z_i)

    progress_bar.update(1)

progress_bar.close()

# find most frequent hyperparameters
best_hparams_reg_treated = models.modes_of_values(hparams_reg_treated)
best_hparams_reg_not_treated = models.modes_of_values(hparams_reg_not_treated)
best_hparams_propensity = models.modes_of_values(hparams_propensity)
if args.balancing_variables is not None:
    best_hparams_reg_Z1 = models.modes_of_values(hparams_reg_Z1)
    best_hparams_reg_Z0 = models.modes_of_values(hparams_reg_Z0)
    best_hparams_propensity_Z = models.modes_of_values(hparams_propensity_Z)
else:
    best_hparams_reg_Z1 = {}
    best_hparams_reg_Z0 = {}
    best_hparams_propensity_Z = {}

results_estimates_DML = None
results_estimates_riesz = None
results_estimates_sample_data = None
seeds_simulations = args.seed + np.arange(args.num_simulations)
# add progress bar
progress_bar = tqdm(total=args.num_simulations, desc="Simulations")


for i in range(args.num_simulations):
    # set seed
    np.random.seed(seeds_simulations[i])

    y, d, z, x = data.simulate_data(
        n=args.n, p=args.p, mode=args.mode,
        balancing_variables=args.balancing_variables, true_effect=False)

    # estimate BGATE

    # using the baseline DML version of us
    results_trajectory_DML = estimators.estimate_effect(
        y, d, z, x,
        model_reg_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_treated),
        model_reg_not_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_not_treated),
        model_propensity=models.classification_model(name=args.model_name, **model_args, **best_hparams_propensity),
        model_reg_treated_z=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_Z1),
        model_reg_not_treated_z=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_Z0),
        model_propensity_z=models.classification_model(name=args.model_name, **model_args, **best_hparams_propensity_Z),
        balancing_variables=args.balancing_variables, nfolds_first=args.n_folds,
        nfolds_second=args.n_folds)

    # create results array
    if results_estimates_DML is None:
        results_estimates_DML = np.zeros(
            (args.num_simulations, *results_trajectory_DML.shape)
        )
    results_estimates_DML[i] = results_trajectory_DML
    
    # using the Riesz DML version with Neural nets

    results_trajectory_riesz = estimators.estimate_effect_riesz(
        y, d, z, x,
        balancing_variables=args.balancing_variables, nfolds_first=args.n_folds,
        nfolds_second=args.n_folds)

    # create results array
    if results_estimates_riesz is None:
        results_estimates_riesz = np.zeros(
            (args.num_simulations, *results_trajectory_riesz.shape)
        )
    results_estimates_riesz[i] = results_trajectory_riesz

    results_trajectory_sample_data = estimators.estimate_effect_data_sampling(
        y, d, z, x,
        model_reg_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_treated),
        model_reg_not_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_not_treated),
        model_propensity=models.classification_model(name=args.model_name, **model_args, **best_hparams_propensity),
        balancing_variables=args.balancing_variables, nfolds_first=args.n_folds)
             
    # create results array
    if results_estimates_sample_data is None:
        results_estimates_sample_data = np.zeros(
            (args.num_simulations, *results_trajectory_sample_data.shape)
        )
    results_estimates_sample_data[i] = results_trajectory_sample_data

    progress_bar.update(1)

progress_bar.close()

# Tune parameters for true effect
# set seed for true effect
np.random.seed(args.seed)

# add progress bar
progress_bar = tqdm(total=10, desc="Hyperparameter tuning")
for i in range(10):
    y, d, z, x = data.simulate_data(
        n=10**5, p=args.p, mode=args.mode,
        balancing_variables=args.balancing_variables, true_effect=False)

    (hparams_reg_treated_i, hparams_reg_not_treated_i, hparams_propensity_i,
     hparams_reg_Z1_i, hparams_reg_Z0_i, hparams_propensity_Z_i) = (
        models.tune_nuisances(y, d, x, z, args.balancing_variables,
                              nfolds=args.n_folds, model_name=args.model_name,
                              n_jobs_cv=args.n_jobs, **model_args,))
    hparams_reg_treated.append(hparams_reg_treated_i)
    hparams_reg_not_treated.append(hparams_reg_not_treated_i)
    hparams_propensity.append(hparams_propensity_i)
    hparams_reg_Z1.append(hparams_reg_Z1_i)
    hparams_reg_Z0.append(hparams_reg_Z0_i)
    hparams_propensity_Z.append(hparams_propensity_Z_i)

    progress_bar.update(1)

progress_bar.close()

# find most frequent hyperparameters
best_hparams_reg_treated_true_effect = models.modes_of_values(hparams_reg_treated)
best_hparams_reg_not_treated_true_effect = models.modes_of_values(hparams_reg_not_treated)
best_hparams_propensity_true_effect = models.modes_of_values(hparams_propensity)
if args.balancing_variables is not None:
    best_hparams_reg_Z1_true_effect = models.modes_of_values(hparams_reg_Z1)
    best_hparams_reg_Z0_true_effect = models.modes_of_values(hparams_reg_Z0)
    best_hparams_propensity_Z_true_effect = models.modes_of_values(hparams_propensity_Z)
else:
    best_hparams_reg_Z1_true_effect = {}
    best_hparams_reg_Z0_true_effect = {}
    best_hparams_propensity_Z_true_effect = {}


# approximate true effect
_, _, _, _, effect_true = data.simulate_data(
    n=10**6, p=args.p, mode=args.mode,
    balancing_variables=args.balancing_variables, true_effect=True,
    njobs=-1,
    model_reg_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_Z1_true_effect),
    model_reg_non_treated=models.regression_model(name=args.model_name, **model_args, **best_hparams_reg_Z0_true_effect),
    model_propensity=models.classification_model(name=args.model_name, **model_args, **best_hparams_propensity_Z_true_effect))

# create results folder if it does not exist
if not os.path.exists("results"):
    os.makedirs("results")
# define file name from input arguments
args_list = list(vars(args).items())
file_name = "__".join([f"{k}{v}" for k, v in args_list])

np.savez(f"results/{file_name}.npz",
         results_estimates_DML=results_estimates_DML,
         results_estimates_riesz=results_estimates_riesz,
         results_estimates_sample_data = results_estimates_sample_data,
         seeds_simulations=seeds_simulations,
         true_effect=effect_true,
         simulation_settings=vars(args),
         best_hparams_reg_treated=best_hparams_reg_treated,
         best_hparams_reg_not_treated=best_hparams_reg_not_treated,
         best_hparams_propensity=best_hparams_propensity,
         best_hparams_reg_Z1=best_hparams_reg_Z1,
         best_hparams_reg_Z0=best_hparams_reg_Z0,
         best_hparams_propensity_Z=best_hparams_propensity_Z,
         best_hparams_reg_treated_true_effect=best_hparams_reg_treated_true_effect,
         best_hparams_reg_not_treated_true_effect=best_hparams_reg_not_treated_true_effect,
         best_hparams_propensity_true_effect=best_hparams_propensity_true_effect,
         best_hparams_reg_Z1_true_effect=best_hparams_reg_Z1_true_effect,
         best_hparams_reg_Z0_true_effect=best_hparams_reg_Z0_true_effect,
         best_hparams_propensity_Z_true_effect=best_hparams_propensity_Z_true_effect,)
