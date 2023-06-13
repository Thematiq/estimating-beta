from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm


class FinancialEstimator:
    def __init__(self, sup_func):
        self._fun = sup_func

    def fit(self, _X, _y) -> FinancialEstimator:
        return self

    def predict(self, X):
        excess_return = (X['stock_yearly_return'] - X['risk_free_return']).values.reshape(-1, 1)
        market_return = X['market_yearly_return'].values.reshape(-1, 1)
        return self._fun(excess_return, market_return)


def eval_beta(df, window, stock='stock_yearly_return', market='market_yearly_return'):
    cov = df[[stock, market]].rolling(window).cov()
    return cov.xs(market, level=1)[stock] / cov.xs(market, level=1)[market]


def optimize_model(
        estimator,
        X, y,
        params_dist,
        opt_samples: int,
        train_splits: int,
        split_size: int = 1,
        scoring=mean_absolute_error):
    n_splits = len(X.index) // split_size
    train_size = train_splits * split_size

    folds = TimeSeriesSplit(max_train_size=train_size, n_splits=n_splits)

    return RandomizedSearchCV(
        estimator,
        param_distributions=params_dist,
        n_iter=opt_samples,
        refit=True,
        cv=folds.split(X),
        scoring=scoring
    ).fit(X, y).best_estimator_


def run_experiment(
        estimator,
        X,
        param_dist,
        opt_samples: int = 100,
        split_size: int = 1,
        cv_outer_splits: int = (5 * 12),
        cv_inner_splits: int = 12,
        beta_corr_window: int = (5 * 12),
        scoring=mean_absolute_error):
    """

    :param estimator:
    :param X: pandas dataframe containing:
        - stock_yearly_return
        - market_yearly_return
        - risk_free_return
    :param param_dist:
    :param opt_samples:
    :param split_size:
    :param cv_outer_splits:
    :param cv_inner_splits:
    :param beta_corr_window:
    :param scoring:
    :return:
    """
    X = X[['stock_yearly_return', 'market_yearly_return', 'risk_free_return']]
    y = eval_beta(X, beta_corr_window)

    n_splits = len(X.index) // split_size
    train_size = split_size * cv_outer_splits

    folds = TimeSeriesSplit(max_train_size=train_size, n_splits=n_splits)

    for train_indices, test_indices in tqdm(folds.split(X)):
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        best_model = optimize_model(
            estimator,
            X_train, y_train,
            param_dist,
            opt_samples,
            cv_inner_splits,
            split_size,
            scoring
        )




