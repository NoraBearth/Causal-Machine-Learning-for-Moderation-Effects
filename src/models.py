import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from src import estimators
from typing import Tuple, List
from collections import Counter


def regression_model(name: str, **kwargs):
    if name == "RandomForest":
        return clone(RandomForestRegressor(n_estimators=500, **kwargs))
    elif name == "GradientBoosting":
        return clone(GradientBoostingRegressor(**kwargs))
    elif name == "Lasso":
        pipeline = Pipeline(
            [("polynomial_features", PolynomialFeatures(degree=2, include_bias=False),),
             ("scaler", StandardScaler()),
             ("regressor", clone(Lasso(**kwargs))),])
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def classification_model(name: str, **kwargs):
    if name == "RandomForest":
        return clone(RandomForestClassifier(n_estimators=500, **kwargs))
    elif name == "GradientBoosting":
        return clone(GradientBoostingClassifier(**kwargs))
    elif name == "Lasso":
        pipeline = Pipeline(
            [("polynomial_features",
              PolynomialFeatures(degree=2, include_bias=False),),
             ("scaler", StandardScaler()),
             ("classifier",
              clone(LogisticRegression(penalty="l1", solver="liblinear", **kwargs)),),])
        return pipeline
    else:
        raise ValueError(f"Model {name} not recognized")


def hyperparameters_grid(name: str, classification: bool = False):
    if name == "RandomForest":
        return {
            "max_depth": [2, 3, 5, 10, 20],
            "min_samples_leaf": [5, 10, 15, 20, 30, 50],
        }
    elif name == "GradientBoosting":
        return {
            "n_estimators": [5, 10, 25, 50, 100, 200, 500],
            "learning_rate": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
            "max_depth": [1, 2, 3, 5, 10],
        }
    elif name == "Lasso":
        if classification:
            return {"classifier__C": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
        else:
            return {"regressor__alpha": [0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 1]}
    else:
        raise ValueError(f"Model {name} not recognized")


def tune_nuisances(y: np.ndarray, d: np.ndarray, x: np.ndarray, z: np.ndarray,
                   balancing_variables: list,
                   model_name: dict, nfolds: int = 2, n_jobs_cv: int = -1,
                   **kwargs,):
    """"Function to tune the optimal hyperparameters"""

    # find optimal hyperparameters
    classification = False
    param_grid = hyperparameters_grid(model_name, classification)
    hyperparameters_reg_treated = _cv(
        np.concatenate((x[d == 1], z[d == 1].reshape(-1, 1)), axis=1),
        y[d == 1], nfolds, regression_model(name=model_name, **kwargs),
        param_grid, n_jobs_cv,)

    hyperparameters_reg_not_treated = _cv(
        np.concatenate((x[d == 0], z[d == 0].reshape(-1, 1)), axis=1),
        y[d == 0], nfolds, regression_model(name=model_name, **kwargs),
        param_grid, n_jobs_cv,)
    classification = True
    param_grid = hyperparameters_grid(model_name, classification)
    hyperparameters_propensity = _cv(
        np.concatenate((x, z.reshape(-1, 1)), axis=1), d, nfolds,
        classification_model(name=model_name, **kwargs), param_grid,
        n_jobs_cv,)

    if balancing_variables is None:

        return (hyperparameters_reg_treated, hyperparameters_reg_not_treated,
                hyperparameters_propensity, None, None, None)

    else:

        hyperparameters_propensity_z = _cv(
            x[:, balancing_variables], z, nfolds,
            classification_model(name=model_name, **kwargs), param_grid,
            n_jobs_cv,)

        score = _get_score(y, d, x, z, nfolds, model_name,
                           hyperparameters_reg_treated,
                           hyperparameters_reg_not_treated,
                           hyperparameters_propensity, **kwargs)

        hyperparameters_reg_Z1 = _cv(
            x[z == 1][:, balancing_variables], score[z == 1], nfolds,
            regression_model(name=model_name, **kwargs), param_grid, n_jobs_cv)

        hyperparameters_reg_Z0 = _cv(
            x[z == 0][:, balancing_variables], score[z == 0], nfolds,
            regression_model(name=model_name, **kwargs), param_grid, n_jobs_cv)

        return (hyperparameters_reg_treated, hyperparameters_reg_not_treated,
                hyperparameters_propensity, hyperparameters_propensity_z,
                hyperparameters_reg_Z1, hyperparameters_reg_Z0)


def _get_score(y, d, x, z, nfolds, model_name, hyperparameters_reg_treated,
               hyperparameters_reg_not_treated, hyperparameters_propensity,
               **kwargs):
    # Initialize data for cross-fitting
    y, d, x, z, idx = estimators._init_estimation(y, d, x, z, nfolds)

    # Initialize arrays for predictions with nan
    y_pred_treated = estimators._nan_array(y.shape)
    y_pred_not_treated = estimators._nan_array(y.shape)
    d_pred = estimators._nan_array(y.shape)

    for i in range(nfolds):
        (y_train, d_train, x_train, z_train), (y_test, d_test, x_test, z_test), idx_test = estimators._get_folds(
            y, d, x, z, idx, i)

        # Predict outcomes using data on the treated
        y_pred_treated[idx_test] = estimators._regression_prediction(
            clone(regression_model(model_name, **kwargs, **hyperparameters_reg_treated)),
            np.concatenate((x_train[d_train == 1, :],
                            z_train[d_train == 1].reshape(-1, 1)), axis=1),
            y_train[d_train == 1],
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1),)

        # Predict outcomes using data on the non-treated
        y_pred_not_treated[idx_test] = estimators._regression_prediction(
            clone(regression_model(model_name, **kwargs, **hyperparameters_reg_not_treated)),
            np.concatenate((x_train[d_train == 0, :],
                            z_train[d_train == 0].reshape(-1, 1)), axis=1),
            y_train[d_train == 0],
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1), )

        # Predict treatment probabilities (propensity score)
        d_pred[idx_test] = estimators._classification_prediction(
            clone(classification_model(model_name, **kwargs, **hyperparameters_propensity)),
            np.concatenate((x_train, z_train.reshape(-1, 1)), axis=1), d_train,
            np.concatenate((x_test, z_test.reshape(-1, 1)), axis=1))

    tau = estimators._compute_tau(y, d, y_pred_treated, y_pred_not_treated,
                                  d_pred, reweight=True)
    return tau


def _cv(x: np.ndarray, y: np.ndarray, nfolds: int, estimator, param_grid: dict,
        n_jobs: int = -1,) -> Tuple[int, int]:
    grid = GridSearchCV(estimator, param_grid, cv=nfolds, n_jobs=n_jobs)
    grid.fit(x, y.ravel())
    best = grid.best_params_
    return {k.split("__")[-1]: v for k, v in best.items()}


def modes_of_values(list_of_dicts: List[dict]) -> dict:
    """
    Given a list of dictionaries, returns the most common value for each key
    """
    keys = list(set().union(*list_of_dicts))
    return {
        key: Counter([d[key] for d in list_of_dicts]).most_common(1)[0][0]
        for key in keys
    }
