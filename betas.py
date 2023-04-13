import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from typing import Optional, Union

# excess_return = portfolio_return - risk_free
# portfolio_return - risk_free = alpha + beta*(market_return - risk_free) [ + beta_f0*factors[0] + beta_f1*factors[1] + ...] 
# factors - e.g. HML, SMB
# risk_free - rate based on government bond rate and inflation rate. Can be found at http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html, as well as factors.

def basic_beta(excess_return : np.array, market_return :np.array) -> np.float64:
    # only without factors
    returns : np.array = np.concatenate([market_return, excess_return], axis=1)
    cov : np.array = np.cov(returns.T)
    return cov[0,-1]/cov[0, 0]

def fama_macbeth_beta(excess_return : np.array, market_return : np.array, factors : Optional[np.array]=None ) -> np.array:
    if factors != None:
        X : np.array = np.concatenate([market_return, factors], axis=1)
    else:
        X: np.array = market_return
    model = LinearRegression().fit(X, excess_return)
    return model.coef_
    
def fama_macbeth_beta_sm(excess_return : np.array, market_return : np.array, factors : Optional[np.array]=None) -> np.array:
    if factors != None:
        X : np.array = np.concatenate([market_return, factors], axis=1)
    else:
        X : np.array = market_return
    X = sm.add_constant(X)
    model = sm.OLS(excess_return, X).fit()
    return model.params[1:]

# def shrinkage_beta(excess_return : np.array, market_return : np.array, factors : Optional[np.array]=None) -> np.array:
#     # betas = fama_macbeth_beta()
#     betas = np.array([1,1,2,3,1,2])
#     var = np.var(betas)
#     prior = betas.mean()
#     print(var, prior)


if __name__ == "__main__":
    portfolio_return = np.array([1,2,3,4]).reshape((-1,1))
    risk_free = np.array([0.1,0.2,0.1,0.1]).reshape((-1,1))
    market_return = np.array([4,3,3,1]).reshape((-1,1))
    factors = np.array([1,1,1,2]).reshape((-1,1))
    
    X = market_return-risk_free
    y = portfolio_return-risk_free
    X = np.concatenate([X, factors], axis=1)
    beta = basic_beta(y, X)
    print(beta)
    beta = fama_macbeth_beta(y, X)
    print(beta)
    beta = fama_macbeth_beta_sm(y, X)
    print(beta)

    # print(shrinkage_beta(y,X))

