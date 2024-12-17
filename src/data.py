import numpy as np
from typing import Tuple
from scipy.stats import beta
from src import estimators

# Define constants
MIN_COVARIATES = 7


def simulate_data(mode=1, n=1000, p=5, balancing_variables=[],
                  true_effect=True, njobs=-1, nfolds=2, model_reg_treated=None,
                  model_reg_non_treated=None, model_propensity=None):

    catalog = {
        1: simulate_linear,
        2: simulate_nonlinear,
        3: simulate_nonlinear_z_influences_x,
    }

    assert mode in catalog, "Invalid mode {}. Should be one of {}".format(
        mode, set(catalog))
    assert p >= MIN_COVARIATES, "Number of covariates should be at least {}".format(
        MIN_COVARIATES)

    return catalog[mode](n, p, balancing_variables, true_effect=true_effect, 
                         njobs=njobs, nfolds=nfolds, model_reg_treated=model_reg_treated,
                         model_reg_non_treated=model_reg_non_treated,
                         model_propensity=model_propensity)


def simulate_linear(n, p, balancing_variables=[], true_effect=False, njobs=-1,
                    nfolds=2, model_reg_treated=None,
                    model_reg_non_treated=None,
                    model_propensity=None,) -> Tuple[
                        np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Synthetic data that is mostly linear
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=7)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - Y ((n,)-array): outcome variable.
            - D ((n,)-array): treatment flag with value 0 or 1.
            - Z ((n,)-array): moderator flag with value 0 or 1.
            - X ((n,p)-ndarray): independent variables.
            - effects (dict): treatment effects
    """

    beta_1 = [0.7, 0.1, 0.7, 0.4, 0.2]
    beta_0 = [0.2, 0.3, 0.6, 0.3, 0.5]

    X = np.full((n, p), np.nan)

    # Create covariates from a uniform and normal distribution
    X[:, 0] = np.random.uniform(0, 1, n)
    X[:, 1] = np.random.uniform(0, 1, n)
    X[:, 2:p] = np.random.normal(0.5, np.sqrt(1/12), (n, p-2))

    f_z = (X[:, 0] * X[:, 1])
    pz = (0.1 + 0.8*beta.cdf(f_z, 2, 4))
    Z = np.random.binomial(1, p=pz, size=n)

    d_x = ((X[:, 0] + X[:, 1] + X[:, 2] + X[:, 5] + Z)/5)

    p_d = (0.2 + 0.6*beta.cdf(d_x, 2, 4))

    D = np.random.binomial(1, p=p_d, size=n)

    # Generate potential outcomes Y(0, 0) and Y(0, 1)

    mu_01 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])
    mu_00 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])

    # Generate treatment effect that is heterogeneous but linear

    tau_1 = (beta_1[0]*X[:, 0] + beta_1[1]*X[:, 1] + beta_1[2]*X[:, 2] +
             beta_1[3]*X[:, 5] + beta_1[4]*Z)
    tau_0 = (beta_0[0]*X[:, 0] + beta_0[1]*X[:, 1] + beta_0[2]*X[:, 2] +
             beta_0[3]*X[:, 5] + beta_0[4]*Z)

    # Generate the potential outcomes Y(1,0) and Y(1, 1)

    mu_11 = mu_01 + tau_1
    mu_10 = mu_00 + tau_0

    # Generate random noise for the potential outcomes

    eps_01 = np.random.normal(loc=0, scale=1, size=n)
    eps_00 = np.random.normal(loc=0, scale=1, size=n)
    eps_11 = np.random.normal(loc=0, scale=1, size=n)
    eps_10 = np.random.normal(loc=0, scale=1, size=n)

    # Generate the potential outcomes

    Y_01 = mu_01 + eps_01
    Y_00 = mu_00 + eps_00
    Y_11 = mu_11 + eps_11
    Y_10 = mu_10 + eps_10

    # Generate the observed outcome

    Y = Y_11*D*Z + Y_01*(1-D)*Z + Y_10*D*(1-Z) + Y_00*(1-D)*(1-Z)

    # Save the true treatment effects
    if true_effect:

        df_effects = estimate_true_effect(
            Y, D, Z, X, model_reg_treated, model_reg_non_treated,
            model_propensity, tau_1, tau_0, balancing_variables,
            nfolds)

        return Y, D, Z, X, df_effects
    else: 
        return Y, D, Z, X


def simulate_nonlinear(n, p, balancing_variables=[], true_effect=False,
                       njobs=-1, nfolds=2, model_reg_treated=None,
                       model_reg_non_treated=None, model_propensity=None
                       ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, dict]:
    """Synthetic data that is nonlinear
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=7)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - Y ((n,)-array): outcome variable.
            - D ((n,)-array): treatment flag with value 0 or 1.
            - Z ((n,)-array): moderator flag with value 0 or 1.
            - X ((n,p)-ndarray): independent variables.
            - effects (dict): treatment effects
    """

    beta_1 = [0.7, 0.1, 0.7, 0.4, 0.2]
    beta_0 = [0.2, 0.3, 0.6, 0.3, 0.5]

    X = np.full((n, p), np.nan)

    # Create covariates from a uniform and normal distribution
    X[:, 0] = np.random.uniform(0, 1, n)
    X[:, 1] = np.random.uniform(0, 1, n)
    X[:, 2:p] = np.random.normal(0.5, np.sqrt(1/12), (n, p-2))

    f_z = (X[:, 0] * X[:, 1])
    pz = (0.1 + 0.8*beta.cdf(f_z, 2, 4))
    Z = np.random.binomial(1, p=pz, size=n)

    d_x = ((X[:, 0] + X[:, 1] + X[:, 2] + X[:, 5] + Z)/5)

    p_d = (0.2 + 0.6*beta.cdf(d_x, 2, 4))

    D = np.random.binomial(1, p=p_d, size=n)

    # Generate potential outcomes Y(0, 0) and Y(0, 1)

    mu_01 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])
    mu_00 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])

    # Generate treatment effect that is heterogeneous and nonlinear

    tau_1 = (np.sin(7*(beta_1[0]*X[:, 0])) + np.sin(20*beta_1[1]*X[:, 1])
             + beta_1[2]*X[:, 2]**4 + beta_1[3]*X[:, 5] + beta_1[4]*Z)

    tau_0 = (np.sin(7*(beta_0[0]*X[:, 0])) + np.sin(20*beta_0[1]*X[:, 1])
             + beta_0[2]*X[:, 2]**2 + beta_0[3]*X[:, 5] + beta_0[4]*Z)

    # Generate the potential outcomes Y(1,0) and Y(1, 1)

    mu_11 = mu_01 + tau_1
    mu_10 = mu_00 + tau_0

    # Generate random noise for the potential outcomes

    eps_01 = np.random.normal(loc=0, scale=1, size=n)
    eps_00 = np.random.normal(loc=0, scale=1, size=n)
    eps_11 = np.random.normal(loc=0, scale=1, size=n)
    eps_10 = np.random.normal(loc=0, scale=1, size=n)

    # Generate the potential outcomes

    Y_01 = mu_01 + eps_01
    Y_00 = mu_00 + eps_00
    Y_11 = mu_11 + eps_11
    Y_10 = mu_10 + eps_10

    # Generate the observed outcome

    Y = Y_11*D*Z + Y_01*(1-D)*Z + Y_10*D*(1-Z) + Y_00*(1-D)*(1-Z)

    # Save the true treatment effects
    if true_effect:

        df_effects = estimate_true_effect(
            Y, D, Z, X, model_reg_treated, model_reg_non_treated,
            model_propensity, tau_1, tau_0, balancing_variables,
            nfolds)

        return Y, D, Z, X, df_effects
    else:
        return Y, D, Z, X


def simulate_nonlinear_z_influences_x(
        n, p, balancing_variables=[], true_effect=False, njobs=-1, nfolds=2,
        model_reg_treated=None, model_reg_non_treated=None,
        model_propensity=None,) -> Tuple[np.ndarray, np.ndarray,
                                         np.ndarray, np.ndarray, dict]:
    """Synthetic data that is nonlinear
    Args:
        n (int, optional): number of observations
        p (int optional): number of covariates (>=7)
        sigma (float): standard deviation of the error term

    Returns:
        (tuple): Synthetically generated samples with the following outputs:
            - Y ((n,)-array): outcome variable.
            - D ((n,)-array): treatment flag with value 0 or 1.
            - Z ((n,)-array): moderator flag with value 0 or 1.
            - X ((n,p)-ndarray): independent variables.
            - effects (dict): treatment effects
    """

    beta_1 = [0.7, 0.1, 0.7, 0.4, 0.2]
    beta_0 = [0.2, 0.3, 0.6, 0.3, 0.5]

    X = np.full((n, p), np.nan)

    # Create covariates from a uniform and normal distribution
    X[:, 0] = np.random.uniform(0, 1, n)
    X[:, 1] = np.random.uniform(0, 1, n)
    X[:, 2:p] = np.random.normal(0.5, np.sqrt(1/12), (n, p-2))

    f_z = (X[:, 0] * X[:, 1])
    pz = (0.1 + 0.8*beta.cdf(f_z, 2, 4))
    Z = np.random.binomial(1, p=pz, size=n)

    px = 1 / (1 + np.exp(-2 * Z))

    X[:, 5] = np.random.binomial(1, p=px, size=n)

    d_x = ((X[:, 0] + X[:, 1] + X[:, 2] + X[:, 5] + Z)/5)

    p_d = (0.2 + 0.6*beta.cdf(d_x, 2, 4))

    D = np.random.binomial(1, p=p_d, size=n)

    # Generate potential outcomes Y(0, 0) and Y(0, 1)

    mu_01 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])
    mu_00 = (np.sin(np.pi * X[:, 0] * X[:, 1]) + (X[:, 2] - 0.5)**2 +
             0.1 * X[:, 3] + 0.3 * X[:, 5])

    # Generate treatment effect that is heterogeneous and nonlinear

    tau_1 = (np.sin(7*(beta_1[0]*X[:, 0])) + np.sin(20*beta_1[1]*X[:, 1])
             + beta_1[2]*X[:, 2]**4 + beta_1[3]*X[:, 5] + beta_1[4]*Z)

    tau_0 = (np.sin(7*(beta_0[0]*X[:, 0])) + np.sin(20*beta_0[1]*X[:, 1])
             + beta_0[2]*X[:, 2]**2 + beta_0[3]*X[:, 5] + beta_0[4]*Z)

    # Generate the potential outcomes Y(1,0) and Y(1, 1)

    mu_11 = mu_01 + tau_1
    mu_10 = mu_00 + tau_0

    # Generate random noise for the potential outcomes

    eps_01 = np.random.normal(loc=0, scale=1, size=n)
    eps_00 = np.random.normal(loc=0, scale=1, size=n)
    eps_11 = np.random.normal(loc=0, scale=1, size=n)
    eps_10 = np.random.normal(loc=0, scale=1, size=n)

    # Generate the potential outcomes

    Y_01 = mu_01 + eps_01
    Y_00 = mu_00 + eps_00
    Y_11 = mu_11 + eps_11
    Y_10 = mu_10 + eps_10

    # Generate the observed outcome

    Y = Y_11*D*Z + Y_01*(1-D)*Z + Y_10*D*(1-Z) + Y_00*(1-D)*(1-Z)

    # Save the true treatment effects
    if true_effect:

        df_effects = estimate_true_effect(
            Y, D, Z, X, model_reg_treated, model_reg_non_treated,
            model_propensity, tau_1, tau_0, balancing_variables,
            nfolds)

        return Y, D, Z, X, df_effects
    else:
        return Y, D, Z, X


def estimate_true_effect(Y, D, Z, X, model_reg_treated, model_reg_non_treated,
                         model_propensity, tau_1, tau_0,
                         balancing_variables, nfolds):

    if balancing_variables is None:
        true_effect = (np.mean(tau_1[Z == 1]) - np.mean(tau_0[Z == 0]))

    else:

        true_effect = estimators.simple_BGATE(
            Y, D, Z, X, model_reg_treated, model_reg_non_treated,
            model_propensity, tau_1, tau_0, balancing_variables, nfolds)

    return true_effect
