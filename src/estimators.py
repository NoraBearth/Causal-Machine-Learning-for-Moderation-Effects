import numpy as np
import pandas as pd
from typing import Tuple
import sklearn.ensemble
from sklearn.base import clone
from itertools import chain, repeat
import torch
from torch.utils.data import DataLoader, TensorDataset
from src import functions_riesznet


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def simple_BGATE(y: np.array, d: np.array, z: np.array, x: np.array,
                 model_reg_treated: sklearn.ensemble,
                 model_reg_not_treated: sklearn.ensemble,
                 model_propensity: sklearn.ensemble,
                 score_Z1: int, score_Z0: int,
                 balancing_variables: list, nfolds: int = 2):

    # initialize data for cross-fitting
    y, d, x, z, idx = _init_estimation(y, d, x, z, nfolds)

    # initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    z_pred = _nan_array(y.shape)
    score = _nan_array(y.shape)

    # loop over folds
    for i in range(nfolds):
        (y_train, d_train, x_train, z_train, score_Z0_train, score_Z1_train), (
            y_test, d_test, x_test, z_test, score_Z0_test, score_Z1_test), idx_test = _get_folds(
            y, d, x, z, idx, i, score_Z0, score_Z1)

        # predict outcomes using data on group 1
        y_pred_treated[idx_test] = _regression_prediction(
            clone(model_reg_treated),
            x_train[z_train == 1][:, balancing_variables],
            score_Z1_train[z_train == 1],
            x_test[:, balancing_variables])

        # predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(
            clone(model_reg_not_treated),
            x_train[z_train == 0][:, balancing_variables],
            score_Z0_train[z_train == 0],
            x_test[:, balancing_variables])

        # predict group probabilities
        z_pred[idx_test] = _classification_prediction(
            clone(model_propensity), x_train[:, balancing_variables],
            z_train, x_test[:, balancing_variables])

    tau = (y_pred_treated - y_pred_not_treated +
           (z/z_pred) * (score_Z1 - y_pred_treated) -
           (1-z)/(1-z_pred) * (score_Z0 - y_pred_not_treated))

    estimate_ate, variance_ate = _get_mean_and_variance(tau)

    return estimate_ate


def estimate_effect(y: np.array, d: np.array, z: np.array, x: np.array,
                    model_reg_treated: sklearn.ensemble,
                    model_reg_not_treated: sklearn.ensemble,
                    model_propensity: sklearn.ensemble,
                    model_reg_treated_z: sklearn.ensemble,
                    model_reg_not_treated_z: sklearn.ensemble,
                    model_propensity_z: sklearn.ensemble,
                    balancing_variables: list, nfolds_first: int,
                    nfolds_second: int):

    if balancing_variables is None:
        estimates_z1 = estimate_ATE(y[z == 1], d[z == 1], x[z == 1],
                                    model_reg_treated, model_reg_not_treated,
                                    model_propensity, nfolds_first)
        estimates_z0 = estimate_ATE(y[z == 0], d[z == 0], x[z == 0],
                                    model_reg_treated, model_reg_not_treated,
                                    model_propensity, nfolds_first)

        estimate_GATE = estimates_z1[:, 0] - estimates_z0[:, 0]
        var_GATE = estimates_z1[:, 1] + estimates_z0[:, 1]
        results = np.stack((estimate_GATE, var_GATE), axis=1)

    else:
        results = estimate_BGATE(y, d, z, x, model_reg_treated,
                                 model_reg_not_treated, model_propensity,
                                 model_reg_treated_z, model_reg_not_treated_z,
                                 model_propensity_z, balancing_variables,
                                 nfolds_first, nfolds_second)

    return results


def estimate_effect_riesz(y: np.array, d: np.array, z: np.array, x: np.array,
                          balancing_variables: list, nfolds_first: int,
                          nfolds_second: int):

    if balancing_variables is None:
        estimates_z1 = estimate_ATE_riesz(
            y[z == 1], d[z == 1], x[z == 1], nfolds_first)
        estimates_z0 = estimate_ATE_riesz(
            y[z == 0], d[z == 0], x[z == 0], nfolds_first)

        estimate_GATE = estimates_z1[:, 0] - estimates_z0[:, 0]
        var_GATE = estimates_z1[:, 1] + estimates_z0[:, 1]
        results = np.stack((estimate_GATE, var_GATE), axis=1)

    else:
        results = estimate_BGATE_riesz(
            y, d, z, x, balancing_variables, nfolds_first, nfolds_second)

    return results


def estimate_effect_data_sampling(
        y: np.array, d: np.array, z: np.array, x: np.array,
        model_reg_treated: sklearn.ensemble,
        model_reg_not_treated: sklearn.ensemble,
        model_propensity: sklearn.ensemble, balancing_variables: list,
        nfolds_first: int):

    if balancing_variables is None:
        estimates_z1 = estimate_ATE(
            y[z == 1], d[z == 1], x[z == 1], model_reg_treated,
            model_reg_not_treated, model_propensity, nfolds_first)
        estimates_z0 = estimate_ATE(
            y[z == 0], d[z == 0], x[z == 0], model_reg_treated,
            model_reg_not_treated, model_propensity, nfolds_first)

        estimate_GATE = estimates_z1[:, 0] - estimates_z0[:, 0]
        var_GATE = estimates_z1[:, 1] + estimates_z0[:, 1]
        results = np.stack((estimate_GATE, var_GATE), axis=1)

    else:
        results = estimate_BGATE_data_sampling(
            y, d, z, x, model_reg_treated,
            model_reg_not_treated,
            model_propensity,
            balancing_variables, nfolds_first)

    return results


def estimate_ATE_riesz(y, d, x, nfolds_first, lambda1=0.1, lambda2=1):

    tau_hat = np.zeros((len(y), 1))

    X = torch.tensor(np.concatenate([d[:,np.newaxis],x], axis=1), dtype=torch.float32).to(device)
    y = torch.tensor(y.reshape(-1,1), dtype=torch.float32).to(device)

    # initialize data for cross-fitting
    y, d, x, idx = _init_estimation(y, d, x, None, nfolds_first)

    for i in range(nfolds_first):
        idx_valid = idx[i]
        idx_train = np.concatenate(idx[:i] + idx[(i+1):])
        X_train, y_train, X_valid, y_valid = X[idx_train], y[idx_train], X[idx_valid], y[idx_valid]

        dataset_train = TensorDataset(X_train,y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

        dataset_valid = TensorDataset(X_valid,y_valid)
        data_loader_valid = DataLoader(dataset_valid, batch_size=64, shuffle=False)

        net = functions_riesznet.RieszNet(X.size(1), dim_common=200, dim_heads=100)
        model = functions_riesznet.RieszModel(net, device, lambda1=lambda1, lambda2=lambda2)
        # fast fit
        model.train(data_loader_train, data_loader_valid, 100, lr=1e-3, patience=2, min_delta=1e-4)
        # fine fit
        model.train(data_loader_train, data_loader_valid, 600, lr=1e-4, patience=10,  min_delta=1e-4)

        tau_hat[idx_valid, :] = np.concatenate([
            model.doubly_robust(X_valid, y_valid).cpu().detach().numpy()], axis=1)

    estimate_ate, variance_ate = _get_mean_and_variance(tau_hat)

    results = np.stack((estimate_ate, variance_ate), axis=1)

    return results


def estimate_BGATE_riesz(y, d, z, x, balancing_variables, nfolds_first,
                         nfolds_second, lambda1=0.1, lambda2=1):

    tau_hat = np.zeros((len(y), 1))
    bgate_hat = np.zeros((len(y), 1))

    X = torch.tensor(np.concatenate([d[:, np.newaxis], z[:, np.newaxis], x], axis=1), dtype=torch.float32).to(device)
    y = torch.tensor(y.reshape(-1, 1), dtype=torch.float32).to(device)

    # Initialize data for cross-fitting (first step)
    y, d, x, idx_first = _init_estimation(y, d, x, None, nfolds_first)

    # Store test indices from the first step
    first_step_test_indices = []

    for i in range(nfolds_first):
        idx_valid = idx_first[i]
        idx_train = np.concatenate(idx_first[:i] + idx_first[(i+1):])
        X_train, y_train, X_valid, y_valid = X[idx_train], y[idx_train], X[idx_valid], y[idx_valid]

        # Store test indices for later second-step processing
        first_step_test_indices.append(idx_valid)

        # Prepare datasets
        dataset_train = TensorDataset(X_train, y_train)
        data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

        dataset_valid = TensorDataset(X_valid, y_valid)
        data_loader_valid = DataLoader(dataset_valid, batch_size=64, shuffle=False)

        # Initialize RieszNet model and train it
        net = functions_riesznet.RieszNet(X.size(1), dim_common=200, dim_heads=100)
        model = functions_riesznet.RieszModel(net, device, lambda1=lambda1, lambda2=lambda2)

        # Fast fit
        model.train(data_loader_train, data_loader_valid, 100, lr=1e-3, patience=2, min_delta=1e-4)
        # Fine fit
        model.train(data_loader_train, data_loader_valid, 600, lr=1e-4, patience=10, min_delta=1e-4)

        # Get predictions for the test fold
        tau_hat[idx_valid, :] = np.concatenate([
            model.doubly_robust(X_valid, y_valid).cpu().detach().numpy()], axis=1)

    # Initialize for second-step cross-fitting (now only use test samples from the first step)
    w = x[:, balancing_variables]
    W = torch.tensor(np.concatenate([z[:, np.newaxis], w], axis=1), dtype=torch.float32).to(device)
    tau = torch.tensor(tau_hat[:, 0].reshape(-1, 1), dtype=torch.float32).to(device)

    for i in range(nfolds_first):
        # Get the test indices from the first step
        idx_first_test = first_step_test_indices[i]

        # Re-split the first-step test samples
        W_first_test, tau_first_test = W[idx_first_test], tau[idx_first_test]
        y_first_test, d_first_test, x_first_test = y[idx_first_test], d[idx_first_test], x[idx_first_test]

        # Re-split the first-step test sample for the second step
        y_first_test, d_first_test, x_first_test, idx_second = _init_estimation(y_first_test, d_first_test, x_first_test, None, nfolds_second)

        for j in range(nfolds_second):
            idx_valid = idx_second[j]
            idx_train = np.concatenate(idx_second[:j] + idx_second[(j+1):])

            W_train, W_valid = W_first_test[idx_train], W_first_test[idx_valid]
            tau_train, tau_valid = tau_first_test[idx_train], tau_first_test[idx_valid]

            # Prepare datasets for second step
            dataset_train = TensorDataset(W_train, tau_train)
            data_loader_train = DataLoader(dataset_train, batch_size=64, shuffle=True)

            dataset_valid = TensorDataset(W_valid, tau_valid)
            data_loader_valid = DataLoader(dataset_valid, batch_size=64, shuffle=False)

            # Initialize RieszNet model and train it
            net = functions_riesznet.RieszNet(W.size(1), dim_common=600, dim_heads=300)
            model = functions_riesznet.RieszModel(net, device, lambda1=lambda1, lambda2=lambda2)

            # Fast fit
            model.train(data_loader_train, data_loader_valid, 300, lr=1e-3, patience=14, min_delta=1e-4)
            # Fine fit
            model.train(data_loader_train, data_loader_valid, 1800, lr=1e-4, patience=70, min_delta=1e-4)

            # Get predictions for the test fold in the second step
            bgate_hat[idx_first_test[idx_valid], :] = np.concatenate([
                model.doubly_robust(W_valid, tau_valid).cpu().detach().numpy()], axis=1)

    # Compute the average treatment effect and variance
    estimate_ate, variance_ate = _get_mean_and_variance(bgate_hat)

    results = np.stack((estimate_ate, variance_ate), axis=1)

    return results


def estimate_ATE(y: np.array, d: np.array, x: np.array,
                 model_reg_treated: sklearn.ensemble,
                 model_reg_not_treated: sklearn.ensemble,
                 model_propensity: sklearn.ensemble, nfolds: int):

    # initialize data for cross-fitting
    y, d, x, idx = _init_estimation(y, d, x, None, nfolds)

    # initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    d_pred = _nan_array(y.shape)

    for i in range(nfolds):
        (y_train, d_train, x_train), (y_test, d_test, x_test), idx_test = _get_folds_ate(
            y, d, x, idx, i)

        # predict outcomes using data on the treated
        y_pred_treated[idx_test] = _regression_prediction(
            clone(model_reg_treated), x_train[d_train == 1, :],
            y_train[d_train == 1], x_test,)

        # predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(
            clone(model_reg_not_treated), x_train[d_train == 0, :],
            y_train[d_train == 0], x_test,)

        # predict treatment probabilities
        d_pred[idx_test] = _classification_prediction(
            clone(model_propensity), x_train, d_train, x_test)

    # tau = _compute_tau(y, d, y_pred_treated, y_pred_not_treated, d_pred)

    tau_reweighted = _compute_tau(
        y, d, y_pred_treated, y_pred_not_treated, d_pred, reweight=True)
    # combine all taus in one array
    # tau = np.concatenate((tau, tau_reweighted), axis=1)
    estimate_ate, variance_ate = _get_mean_and_variance(tau_reweighted)

    # metrics is a matrix with rows corresponding to the different approaches
    results = np.stack((estimate_ate, variance_ate), axis=1)

    return results


def estimate_BGATE(y: np.array, d: np.array, z: np.array, x: np.array,
                   model_reg_treated: sklearn.ensemble,
                   model_reg_not_treated: sklearn.ensemble,
                   model_propensity: sklearn.ensemble,
                   model_reg_treated_z: sklearn.ensemble,
                   model_reg_not_treated_z: sklearn.ensemble,
                   model_propensity_z: sklearn.ensemble,
                   balancing_variables: list, nfolds_first: int,
                   nfolds_second: int):

    # Initialize data for cross-fitting
    y, d, x, z, idx = _init_estimation(y, d, x, z, nfolds_first)
    # Initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    d_pred = _nan_array(y.shape)

    # To store first-step test indices for second step processing
    first_step_test_indices = []

    # First Step: Cross-fitting
    for i in range(nfolds_first):
        (y_train, d_train, x_train, z_train), (y_test, d_test, x_test, z_test), idx_test = _get_folds_bgate(
            y, d, x, z, idx, i)

        first_step_test_indices.append(idx_test)  # Store test indices from the first step

        # Predict outcomes using data on the treated
        y_pred_treated[idx_test] = _regression_prediction(
            clone(model_reg_treated),
            np.concatenate((x_train[d_train == 1, :],
                            z_train[d_train == 1].reshape(-1, 1)), axis=1),
            y_train[d_train == 1],
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1))

        # Predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(
            clone(model_reg_not_treated),
            np.concatenate((x_train[d_train == 0, :],
                            z_train[d_train == 0].reshape(-1, 1)), axis=1),
            y_train[d_train == 0],
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1))

        # Predict treatment probabilities (propensity score)
        d_pred[idx_test] = _classification_prediction(
            clone(model_propensity),
            np.concatenate((x_train, z_train.reshape(-1, 1)), axis=1), d_train,
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1))

    tau_reweighted = _compute_tau(
        y, d, y_pred_treated, y_pred_not_treated, d_pred, reweight=True)

    # Initialize second-step predictions
    tau_Z1_pred = _nan_array(y.shape)
    tau_Z0_pred = _nan_array(y.shape)
    z_pred = _nan_array(y.shape)

    # Second Step: Only operate on first step test samples
    for i in range(nfolds_first):  # Loop through the first step folds again

        idx_first_test = first_step_test_indices[i]  # Get the test indices from the first step

        # Now re-split the test set from the first step into second-step folds
        y_first_test, d_first_test, x_first_test, z_first_test, tau_first_test = (
            y[idx_first_test], d[idx_first_test], x[idx_first_test],
            z[idx_first_test], tau_reweighted[idx_first_test])

        y_first_test, d_first_test, x_first_test, z_first_test, idx_second = _init_estimation(y_first_test, d_first_test, x_first_test, z_first_test, nfolds_second)

        # Loop through each fold in the second step
        for j in range(nfolds_second):
            (y_train2, d_train2, x_train2, z_train2, tau_train2), (
                y_test2, d_test2, x_test2, z_test2, tau_test2), idx_test2 = _get_folds_bgate(
                y_first_test, d_first_test, x_first_test, z_first_test, idx_second, j, tau_first_test)

            # Train second step model for Z=1
            tau_Z1_pred[idx_first_test[idx_test2]] = _regression_prediction(
                clone(model_reg_treated_z),
                x_train2[z_train2 == 1][:, balancing_variables],
                tau_train2[z_train2 == 1],
                x_test2[:, balancing_variables],)

            # Train second step model for Z=0
            tau_Z0_pred[idx_first_test[idx_test2]] = _regression_prediction(
                clone(model_reg_not_treated_z),
                x_train2[z_train2 == 0][:, balancing_variables],
                tau_train2[z_train2 == 0],
                x_test2[:, balancing_variables],)

            # Predict group probabilities (propensity score)
            z_pred[idx_first_test[idx_test2]] = _classification_prediction(
                clone(model_propensity_z),
                x_train2[:, balancing_variables], z_train2,
                x_test2[:, balancing_variables])

    tau_bgate = _compute_tau(
        tau_reweighted, z, tau_Z1_pred, tau_Z0_pred, z_pred, reweight=False)

    tau_reweighted_bgate = _compute_tau(
        tau_reweighted, z, tau_Z1_pred, tau_Z0_pred, z_pred, reweight=True)

    # Combine all taus in one array
    tau_bgate = np.concatenate((tau_bgate, tau_reweighted_bgate), axis=1)

    estimate_bgate, variance_bgate = _get_mean_and_variance(tau_bgate)

    # Metrics is a matrix with rows corresponding to the different approaches
    results = np.stack((estimate_bgate, variance_bgate), axis=1)

    return results


def estimate_BGATE_data_sampling(y: np.array, d: np.array, z: np.array,
                                 x: np.array,
                                 model_reg_treated: sklearn.ensemble,
                                 model_reg_not_treated: sklearn.ensemble,
                                 model_propensity: sklearn.ensemble,
                                 balancing_variables,
                                 nfolds: int):

    # Replace the original balancing variables with the newly drawn random values
    non_balancing_indices = [i for i in range(x.shape[1]) if i not in balancing_variables]
    data_df = pd.DataFrame(np.concatenate([y.reshape(-1,1), d.reshape(-1,1), 
                                           z.reshape(-1,1), x[:, balancing_variables], x[:, non_balancing_indices]], axis=1))

    # Create new column names
    new_column_names = ['y_variable']  # Start with the y_variable name
    new_column_names += ['d_variable']  # Start with the d_variable name
    new_column_names += ['z_variable']  # Start with the z_variable name

    # Add w_n names for the columns in x
    new_column_names += [f'w_{i+1}' for i in range(x[:, balancing_variables].shape[1])]

    # Add x_n names for the columns in other_x
    new_column_names += [f'x_{i+1}' for i in range(x[:, non_balancing_indices].shape[1])]

    data_df.columns = new_column_names

    bgate_name = [col for col in data_df.columns if col.startswith('w')]

    data_new_df = ref_data_bgate(data_df, 'z_variable', bgate_name, 'd_variable')

    # shuffle the data, as it is now sorted by z
    data_new_df = data_new_df.sample(frac=1, random_state=42).reset_index(drop=True)

    data_new_df['weight'] = data_new_df.groupby(data_new_df.columns.tolist()).transform('size')
    data_new_df['weight'] = data_new_df.groupby(data_new_df.columns.tolist())['weight'].transform('max')

    y_new = data_new_df['y_variable'].to_numpy()
    d_new = data_new_df['d_variable'].to_numpy()
    z_new = data_new_df['z_variable'].to_numpy()
    x_new = data_new_df.drop(columns=['y_variable', 'd_variable', 'z_variable', 'weight']).to_numpy()
    weight = data_new_df['weight'].to_numpy()

    ATE_Z1 = estimate_weighted_ATE(
        y_new[z_new == 1], d_new[z_new == 1], x_new[z_new == 1],
        weight[z_new == 1], model_reg_treated, model_reg_not_treated,
        model_propensity, nfolds)

    ATE_Z0 = estimate_weighted_ATE(
        y_new[z_new == 0], d_new[z_new == 0], x_new[z_new == 0],
        weight[z_new == 0], model_reg_treated, model_reg_not_treated,
        model_propensity, nfolds)

    estimate_ate = ATE_Z1[0][0] - ATE_Z0[0][0]
    variance_ate = ATE_Z1[0][1] + ATE_Z0[0][1]

    estimate_ate = np.array([estimate_ate])  # Convert to an array
    variance_ate = np.array([variance_ate])  # Convert to an array

    # Now stack them along the first axis (vertically)
    results = np.stack((estimate_ate, variance_ate), axis=1)

    return results


def estimate_weighted_ATE(y: np.array, d: np.array, x: np.array,
                          weights: np.array,
                          model_reg_treated: sklearn.ensemble,
                          model_reg_not_treated: sklearn.ensemble,
                          model_propensity: sklearn.ensemble, nfolds: int):

    # initialize data for cross-fitting
    y, d, x, idx = _init_estimation(y, d, x, None, nfolds)

    # initialize arrays for predictions with nan
    y_pred_treated = _nan_array(y.shape)
    y_pred_not_treated = _nan_array(y.shape)
    d_pred = _nan_array(y.shape)

    for i in range(nfolds):
        (y_train, d_train, x_train), (y_test, d_test, x_test), idx_test = _get_folds_ate(
            y, d, x, idx, i)

        # predict outcomes using data on the treated
        y_pred_treated[idx_test] = _regression_prediction(
            clone(model_reg_treated), x_train[d_train == 1, :],
            y_train[d_train == 1], x_test,)

        # predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = _regression_prediction(
            clone(model_reg_not_treated), x_train[d_train == 0, :],
            y_train[d_train == 0], x_test,)

        # predict treatment probabilities
        d_pred[idx_test] = _classification_prediction(
            clone(model_propensity), x_train, d_train, x_test)

    # tau = _compute_tau(y, d, y_pred_treated, y_pred_not_treated, d_pred)

    tau_reweighted = _compute_tau(
        y, d, y_pred_treated, y_pred_not_treated, d_pred, reweight=True)

    # Compute the weighted mean and variance of tau
    estimate_ate, variance_ate = _get_mean_and_variance_weighted(
        tau_reweighted, weights)

    # Combine the results into a matrix
    results = np.stack((estimate_ate, variance_ate), axis=1)

    return results


def ref_data_bgate(data_df, z_name, bgate_name, d_name):
    """Create reference samples for covariates (BGATE)."""

    obs = len(data_df)
    idx = np.arange(obs)

    new_idx_dataframe = list(chain.from_iterable(repeat(idx, 2)))
    data_new_df = data_df.loc[new_idx_dataframe, :]
    new_values_z = list(chain.from_iterable([[i] * obs for i in [0, 1]]))
    data_new_df.loc[:, z_name] = new_values_z

    data_new_b_np = data_new_df[bgate_name].to_numpy(copy=True)
    data_new_z_np = data_new_df[z_name].to_numpy(copy=True)
    data_org_b_np = data_df[bgate_name].to_numpy()
    data_org_z_np = data_df[z_name].to_numpy()
    data_org_np = data_df.to_numpy()
    if data_org_b_np.shape[1] > 1:
        bz_cov_inv = invcovariancematrix(data_org_b_np)

    for idx, z_value in enumerate(data_new_z_np):
        z_true = data_org_z_np == z_value

        data_org_np_condz = data_org_np[z_true]
        data_org_b_np_condz = data_org_b_np[z_true]
        diff = data_org_b_np_condz - data_new_b_np[idx, :]
        if data_org_b_np.shape[1] > 1:
            dist = np.sum((diff @ bz_cov_inv) * diff, axis=1)
        else:
            dist = diff**2
        match_neighbour_idx = np.argmin(dist)
        data_new_df.iloc[idx] = data_org_np_condz[match_neighbour_idx, :]

    return data_new_df


def invcovariancematrix(data_np):
    """Compute inverse of covariance matrix and adjust for missing rank."""
    k = np.shape(data_np)
    if k[1] > 1:
        cov_x = np.cov(data_np, rowvar=False)
        rank_not_ok, counter = True, 0
        while rank_not_ok:
            if counter == 20:
                cov_x *= np.eye(k[1])
            if counter > 20:
                cov_inv = np.eye(k[1])
                break
            if np.linalg.matrix_rank(cov_x) < k[1]:
                cov_x += 0.5 * np.diag(cov_x) * np.eye(k[1])
                counter += 1
            else:
                cov_inv = np.linalg.inv(cov_x)
                rank_not_ok = False
    return cov_inv


def _get_mean_and_variance(x: np.ndarray) -> Tuple[float, float]:

    return (np.nanmean(x, axis=0), np.nanvar(x, axis=0) / np.sum(~np.isnan(x), axis=0))


def _get_mean_and_variance_weighted(x: np.ndarray, weights: np.ndarray
                                    ) -> Tuple[float, float]:

    # Variance
    variance = np.nanvar(x, axis=0) / np.sum(~np.isnan(x), axis=0)

    # Covariance due to duplicates

    x_duplicates = x[weights > 1]
    weights_duplicates = weights[weights > 1]
    x_unique, indices = np.unique(x_duplicates, return_index=True)
    weights_unique = weights_duplicates[indices]

    covariance = (np.nansum(weights_unique.reshape(-1, 1)*(
        weights_unique - 1).reshape(-1, 1) *
        (x_unique.reshape(-1, 1) - np.nanmean(x, axis=0))**2, axis=0) /
        np.sum(~np.isnan(x), axis=0)**2)

    return (np.nanmean(x, axis=0), variance + covariance)


def _regression_prediction(
        model: sklearn.ensemble, x_train: np.ndarray, y_train: np.ndarray,
        x_test: np.ndarray) -> np.ndarray:

    model.fit(x_train, y_train.ravel())
    # predict outcomes
    y_pred = model.predict(x_test)
    return y_pred


def _classification_prediction(
        model: sklearn.ensemble, x_train: np.ndarray, w_train: np.ndarray,
        x_test: np.ndarray) -> np.ndarray:

    model.fit(x_train, w_train.ravel())

    # predict treatment probabilities
    w_pred = model.predict_proba(x_test)[:, 1]
    return w_pred


def _init_estimation(y: np.ndarray, w: np.ndarray, x: np.ndarray,
                     z: np.ndarray = None, nfolds: int = 2) -> np.ndarray:

    n = x.shape[0]
    idx = np.random.choice(np.arange(n), size=n, replace=False)
    idx = np.array_split(idx, nfolds)

    if z is None:
        return y, w, x, idx
    else:
        return y, w, x, z, idx


def _get_folds_ate(y: np.ndarray, w: np.ndarray, x: np.ndarray,
                   fold_indices: np.array, fold_idx: int):

    # Split sample into train and test indices
    idx_test = fold_indices[fold_idx]
    idx_train = np.concatenate(fold_indices[:fold_idx] + fold_indices[(fold_idx + 1):])

    # Assign training and test sets for y, w, and x
    x_train, y_train, w_train = x[idx_train], y[idx_train], w[idx_train]
    x_test, y_test, w_test = x[idx_test], y[idx_test], w[idx_test]

    # Base return values (y, w, and x)
    train_data = (y_train, w_train, x_train)
    test_data = (y_test, w_test, x_test)

    return train_data, test_data, idx_test


def _get_folds_bgate(y: np.ndarray, w: np.ndarray, x: np.ndarray,
                     z: np.ndarray, fold_indices: np.array, fold_idx: int,
                     tau: np.ndarray=None):

    # Split sample into train and test indices
    idx_test = fold_indices[fold_idx]
    idx_train = np.concatenate(fold_indices[:fold_idx] + fold_indices[(fold_idx + 1):])

    # Assign training and test sets for y, w, and x
    x_train, y_train, w_train, z_train = x[idx_train], y[idx_train], w[idx_train], z[idx_train]
    x_test, y_test, w_test, z_test = x[idx_test], y[idx_test], w[idx_test], z[idx_test]

    # Base return values (y, w, and x)
    train_data = (y_train, w_train, x_train, z_train)
    test_data = (y_test, w_test, x_test, z_test)

    # Add z if available
    if tau is not None:
        tau_train, tau_test = tau[idx_train], tau[idx_test]
        train_data += (tau_train,)
        test_data += (tau_test,)

    return train_data, test_data, idx_test


def _get_folds(y: np.ndarray, w: np.ndarray, x: np.ndarray, z: np.ndarray,
               fold_indices: np.array, fold_idx: int,
               score_Z0: np.ndarray = None, score_Z1: np.ndarray = None):

    # Split sample into train and test indices
    idx_test = fold_indices[fold_idx]
    idx_train = np.concatenate(fold_indices[:fold_idx] + fold_indices[(fold_idx + 1):])

    # Assign training and test sets for y, w, and x
    x_train, y_train, w_train = x[idx_train], y[idx_train], w[idx_train]
    x_test, y_test, w_test = x[idx_test], y[idx_test], w[idx_test]

    # Base return values (y, w, and x)
    train_data = (y_train, w_train, x_train)
    test_data = (y_test, w_test, x_test)

    # Add z if available
    if z is not None:
        z_train, z_test = z[idx_train], z[idx_test]
        train_data += (z_train,)
        test_data += (z_test,)

    # Add score_Z0 if available
    if score_Z0 is not None:
        score_Z0_train, score_Z0_test = score_Z0[idx_train], score_Z0[idx_test]
        train_data += (score_Z0_train,)
        test_data += (score_Z0_test,)

    # Add score_Z1 if available
    if score_Z1 is not None:
        score_Z1_train, score_Z1_test = score_Z1[idx_train], score_Z1[idx_test]
        train_data += (score_Z1_train,)
        test_data += (score_Z1_test,)

    return train_data, test_data, idx_test


def _nan_array(shape: Tuple[int, int]) -> np.ndarray:
    nan_array = np.full(shape, np.nan, dtype=float)
    return nan_array


def _add_second_axis(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        x = x[:, np.newaxis]
    return x


def _compute_tau(y: np.array, w: np.array, y_pred_treated: np.array,
                 y_pred_not_treated: np.array, w_pred: np.array,
                 reweight: bool = False, tol_propensity: float = 1e-10,):
    y = _add_second_axis(y)
    w = _add_second_axis(w)
    y_pred_treated = _add_second_axis(y_pred_treated)
    y_pred_not_treated = _add_second_axis(y_pred_not_treated)
    w_pred = _add_second_axis(w_pred)

    # winsorize propensity scores
    w_pred = np.minimum(np.maximum(w_pred, tol_propensity), 1 - tol_propensity)

    # define weights
    weight_treated = w / w_pred
    weight_not_treated = (1 - w) / (1 - w_pred)

    # normalize weights
    if reweight:
        weight_treated = weight_treated / np.nanmean(weight_treated, axis=0)
        weight_not_treated = weight_not_treated / np.nanmean(weight_not_treated, axis=0)

    tau = (y_pred_treated - y_pred_not_treated +
           weight_treated * (y - y_pred_treated) -
           weight_not_treated * (y - y_pred_not_treated))
    return tau
