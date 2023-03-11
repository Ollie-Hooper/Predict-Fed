from sklearn.metrics import r2_score, mean_squared_error as MSE

from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.ensemble import RandomForest, XGBoost
from predict_fed.pipeline import Pipeline
from predict_fed.tuning import tune_model
from predict_fed.plotting import plot_pred, rounded_scatter

xg_grid = dict(
    max_depth=[i + 1 for i in range(10)],
    n_estimators=[10, 100, 1000],
    learning_rate=[0.1, 0.2, 0.3],
    colsample_bytree=[0.2, 0.4, 0.6, 0.8, 1],
    subsample=[0.2, 0.4, 0.6, 0.8, 1],
)

rf_grid = dict(
    max_depth=[i + 1 for i in range(10)],
    n_estimators=[10, 100, 1000],
    min_samples_split=[2, 4, 6, 8],
    min_samples_leaf=[1, 2, 3, 4],
    max_features=[0.2, 0.4, 0.6, 0.8, 1.0]
)

data_params = dict(
    smote=True,
    normalisation=True,
)


def main():
    model = RandomForest  # or XGBoost
    grid = rf_grid  # or xg_grid

    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.PoP_PCT_CHANGE_ANN, Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    best_params = tune_model(model=model, y=rate, features=features, grid=grid, search='grid', pipe_params=data_params)

    best_model = model(model_params=best_params)

    pipe = Pipeline(y=rate, features=features, model=best_model, **data_params)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    y_pred, y_pred_rounded = pipe.predict(X_test)
    y_pred_valid, y_pred_rounded_valid = pipe.predict(X_valid)

    print("Valid MSE:", MSE(y_valid, y_pred_valid))
    print("Valid r^2:", r2_score(y_valid, y_pred_valid))
    print("Test MSE:", MSE(y_test, y_pred))
    print("Test r^2:", r2_score(y_test, y_pred))

    plot_pred(y_pred_valid, y_pred_rounded_valid, y_valid)
    rounded_scatter(y_pred_rounded_valid, y_valid)


if __name__ == '__main__':
    main()
