import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import statsmodels.api as sm
from typing import Optional, Union

# excess_return = portfolio_return - risk_free
# fama-macbeth:
# portfolio_return - risk_free = alpha + beta*(market_return - risk_free) [ + beta_f0*factors[0] + beta_f1*factors[1] + ...] 
# factors - e.g. HML, SMB
# risk_free - rate based on government bond rate and inflation rate. Can be found at http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, as well as factors.

def basic_beta(excess_return : np.array, market_return :np.array) -> np.array:
    # only without factors
    returns : np.array = np.concatenate([market_return, excess_return], axis=1)
    cov : np.array = np.cov(returns.T)
    return np.array(cov[0,-1]/cov[0, 0])

def fama_macbeth_beta(excess_return : np.array, market_return : np.array) -> np.array:
    model = LinearRegression().fit(market_return, excess_return)
    return model.coef_[0]
    
def fama_macbeth_beta_sm(excess_return : np.array, market_return : np.array) -> np.array:
    X = sm.add_constant(market_return)
    model = sm.OLS(excess_return, X).fit()
    return model.params[1:]


def shrinkage_beta(excess_return : np.array, market_return : np.array) -> np.array:
    # using Ridge regression via https://en.wikipedia.org/wiki/Shrinkage_(statistics)
    model = Ridge(alpha=1.0).fit(market_return, excess_return)
    return model.coef_[0]

def shrinkage_beta_sm(excess_return : np.array, market_return : np.array) -> np.array:
    # using Ridge regression
    X = sm.add_constant(market_return)
    model = sm.OLS(excess_return, X).fit_regularized(L1_wt=0, alpha=1.0)
    return model.params[1:]


def dimson_betas(excess_return : np.array, market_return : np.array, N=1) -> np.array:
    #without factors
    lagged_market_return = []
    lagged_market_return.append(np.concatenate([market_return[0], np.zeros((market_return.shape[1],2))], axis=1))
    for i in range(1, market_return.shape[0]):
        if N==1:
            second_col = np.zeros((market_return.shape[1],1))
        else:
            second_col = market_return[np.max((0,i-N)):i-1].sum(axis=0)
        lagged_market_return.append(np.concatenate([market_return[i], market_return[i-1], second_col], axis=1))
    

    return np.array([fama_macbeth_beta(er,mr) for er,mr in zip(excess_return, np.array(lagged_market_return))]).sum(axis=1)

def dimson_beta(excess_return : np.array, market_return : np.array) -> np.array:
    return np.array([fama_macbeth_beta(excess_return[:lag+1],market_return[:lag+1]) for lag in range(excess_return.shape[0])]).mean(axis=0)

def get_betas(excess_return: np.array, market_return: np.array, f, **kw) -> np.array:
    return np.array([f(er, mr, **kw) for er, mr in zip(excess_return, market_return)]).reshape(-1)


if __name__ == "__main__":
    # computing singular beta
    portfolio_return = np.array([1,2,3,4]).reshape((-1,1))
    risk_free = np.array([0.1,0.2,0.1,0.1]).reshape((-1,1))
    market_return = np.array([4,3,3,1]).reshape((-1,1))
    factors = np.array([1,1,1,2]).reshape((-1,1))


    X = market_return-risk_free
    y = portfolio_return-risk_free
    # X = np.concatenate([X, factors], axis=1)
    beta = basic_beta(y, market_return-risk_free)
    print("Basic Beta ", beta)
    beta = fama_macbeth_beta(y, X)
    print("FamaMacbeth Beta ",beta)
    beta = fama_macbeth_beta_sm(y, X)
    print("FamaMacbeth Beta (statmodels) ", beta)
    beta = shrinkage_beta(y,X)
    print("Shrinkage Beta ",beta)
    beta = shrinkage_beta_sm(y,X)
    print("Shrinkage Beta (statmodels) ", beta)
    beta = dimson_beta(y,X)
    print("Dimson Beta", beta)
    
    # computing series of beta
    portfolio_return = np.array([[1,2,1,4],[2,1,4,5],[3,4,4,1],[3,4,4,1]])
    portfolio_return = portfolio_return.reshape(portfolio_return.shape[0],-1, 1)

    risk_free = np.array([[0.1,0.2,0.1,0.1],[0.1,0.2,0.1,0.1],[0.1,0.2,0.1,0.1],[0.1,0.2,0.1,0.1]])
    risk_free = risk_free.reshape(risk_free.shape[0], -1, 1)

    market_return = np.array([[4,3,3,1],[4,3,9,1],[4,1,3,1],[5,1,3,1]])
    market_return= market_return.reshape(market_return.shape[0],-1, 1)

    factors = np.array(list(map(lambda x: x.T, np.array([[[4,3,3,1],[4,3,3,1]],[[4,3,3,1],[4,3,3,1]],[[4,3,3,1],[4,3,3,1]],[[4,3,3,1],[4,3,3,1]]]))))


    X = market_return - risk_free
    # X = np.concatenate([X, factors], axis=2)
    y = portfolio_return - risk_free

    betas = get_betas(y, X, fama_macbeth_beta)
    print("FamaMacbeth Beta series ", betas)
    betas = dimson_betas(y,X,3)
    print("Dimson Beta series ",betas)
