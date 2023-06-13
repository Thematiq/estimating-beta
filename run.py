from experiment import run_experiment, build_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


if __name__ == '__main__':
    grid = {
        'n_estimators': [100, 200]
    }

    model = RandomForestRegressor(random_state=42)

    data = build_data(
        stock='GOOG',
        market='^spx',
        risk_free='risk_free_us.csv'
    )

    run_experiment(
        estimator=model,
        data=data,
        param_dist=grid,
        input_columns=['beta'],
        lag_size=21,
        split_size=5,
        opt_samples=2,
        scoring=r2_score
    )
