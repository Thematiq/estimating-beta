from __future__ import annotations

import os

import numpy as np
import pandas as pd

from time import time
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
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
        split_size,
        scoring,
        greater_is_better):
    n_splits = (len(X.index) // split_size) - 1
    train_size = train_splits * split_size

    folds = TimeSeriesSplit(max_train_size=train_size, n_splits=n_splits)

    return RandomizedSearchCV(
        estimator,
        param_distributions=params_dist,
        n_iter=opt_samples,
        refit=True,
        cv=folds.split(X),
        scoring=make_scorer(scoring, greater_is_better=greater_is_better)
    ).fit(X, y).best_estimator_


def build_x_y(data, input_columns, lag_size, y_col='beta'):
    X = pd.DataFrame(index=data.index)
    X['y'] = data[y_col]

    cols = []

    for idx, col_name in enumerate(input_columns):
        for lag_idx in range(1, lag_size+1):
            col = data[col_name].shift(lag_idx)
            cols.append(col)

    return pd.concat([X] + cols, axis=1)


def build_data(market, stock, risk_free, root='data', value_source='Otwarcie'):
    stock_path = os.path.join(root, market, 'history', f'{stock}.csv')
    market_path = os.path.join(root, market, 'history', f'{market}.csv')
    risk_free_path = os.path.join(root, risk_free)

    market_df = pd.read_csv(market_path)
    market_df['value'] = market_df[value_source]
    market_df['date'] = pd.to_datetime(market_df['Data']).dt.date

    stock_df = pd.read_csv(stock_path)
    stock_df['value'] = stock_df[value_source]
    stock_df['date'] = pd.to_datetime(stock_df['Data']).dt.date

    rf_df = pd.read_csv(risk_free_path)
    rf_df['date'] = pd.to_datetime(rf_df['date']).dt.date

    data = market_df[['value', 'date']].merge(stock_df[['value', 'date']], on='date',
                                              suffixes=('_market', '_stock')).merge(
        rf_df[['risk_free_return', 'date']], on='date')

    stock_shift = data['value_stock'].shift(periods=365)
    market_shift = data['value_market'].shift(periods=365)

    data['stock_yearly_return'] = (data['value_stock'] - stock_shift) / stock_shift
    data['market_yearly_return'] = (data['value_market'] - market_shift) / market_shift
    data['excess_return'] = data['stock_yearly_return'] - data['risk_free_return']

    return data


def run_experiment(
        estimator,
        data,
        param_dist,
        input_columns=['beta'],
        lag_size: int = 365 * 5,
        opt_samples: int = 100,
        split_size: int = 1,
        cv_outer_splits: int = (5 * 12),
        cv_inner_splits: int = 12,
        beta_corr_window: int = (5 * 12),
        scoring=mean_absolute_error,
        greater_is_better=True):
    """

    :param estimator: Scikit-learn interface estimator
    :param X: pandas dataframe containing:
        - stock_yearly_return
        - market_yearly_return
        - risk_free_return
    :param param_dist: hyperparameter distribution grid for optimization
    :param opt_samples: number of samples in inner CV optimization
    :param split_size: number of elements in a single split
    :param cv_outer_splits: number of splits in outer training data
    :param cv_inner_splits: number of splits in inner training data
    :param beta_corr_window: beta calculation window
    :param scoring: sklearn scoring function
    :param greater_is_better: If true, higher metric is better
    :return:
    """
    scores = []

    data = data[['stock_yearly_return', 'market_yearly_return', 'risk_free_return', 'date']]
    data.loc[:, 'beta'] = eval_beta(data, beta_corr_window)
    X = build_x_y(data, input_columns, lag_size, 'beta')
    X = X.dropna()
    X, y = X.drop(columns='y'), X['y']

    n_splits = (len(X.index) // split_size) - 1
    train_size = split_size * cv_outer_splits

    folds = TimeSeriesSplit(max_train_size=train_size, n_splits=n_splits)

    for train_indices, test_indices in tqdm(folds.split(X)):
        if len(train_indices) < train_size:
            continue

        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]

        print(f'Running fold {X_train.index.min()} - {X_test.index.max()}')
        start = time()

        best_model = optimize_model(
            estimator,
            X_train, y_train,
            param_dist,
            opt_samples,
            cv_inner_splits,
            split_size,
            scoring,
            greater_is_better
        )

        y_pred = best_model.predict(X_test)
        scores.append(scoring(y_test, y_pred))

        end = time()

        print(f'Fold score: {scores[-1]:.2}, time: {end - start:.2}s')

    return np.mean(scores), np.std(scores)
