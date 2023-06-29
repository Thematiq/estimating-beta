from experiment import run_experiment, build_data, run_tests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from betas import fama_macbeth_beta, shrinkage_beta, dimson_beta
from esn import ESN, WeightInitializer, CompositeInitializer, A
from prophet import Prophet

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':

    i = CompositeInitializer()\
    .with_seed(23)\
    # .sparse(density=0.4)\
    # .spectral_noisy(spectral_radius=1.5, noise_magnitude=0.1)\
    # .uniform()\
    # .regular_graph(4)\
    # .scale(0.9)\

    INPUT_SIZE = 365
    
    esn_initializer = WeightInitializer()
    esn_initializer.weight_hh_init = i

    data = build_data(
        stock='GOOG',
        market='^spx',
        risk_free='risk_free_us.csv'
    )
    betas = [fama_macbeth_beta, shrinkage_beta, dimson_beta]
    estimators =  [
        # [
        #     LinearRegression(),
        #     {
        #         'fit_intercept' : [True]
        #     }
        # ],
        # [
        #     RandomForestRegressor( random_state=42),
        #     {
        #     'n_estimators': [10, 100],
        #     'max_depth': [5,10]
        #     }
        # ],
        [
            ESN(input_size=INPUT_SIZE, initializer=esn_initializer),
            {
            'input_size': [INPUT_SIZE],
            'hidden_size': [500],
            'output_dim': [1],
            'bias': [False],
            'initializer': [esn_initializer],
            'num_layers': [1],
            'activation': [A.self_normalizing_default()],
            'washout': [0],
            'regularization': [1]
            }
        ],
        # [
        #     Prophet(),
        #     {}
        # ]
        # [
        #     XGBRegressor(),
        #     {
        #     'n_estimators': [10, 100],
        #     'max_depth': [5,10]
        #     }
        # ]
        ]
    scorings = [[r2_score, True]]

    run_tests(betas,
        estimators,
        scorings,
        data,
        input_columns=['beta'],
        lag_size=INPUT_SIZE,
        split_size=7,
        opt_samples=3,
        limit_outer_folds=10)


    # run_experiment(
    #     estimator=model,
    #     data=data,
    #     param_dist=grid,
    #     input_columns=['beta'],
    #     lag_size=21,
    #     split_size=5,
    #     opt_samples=1,
    #     # scoring=r2_score
    # )

    
