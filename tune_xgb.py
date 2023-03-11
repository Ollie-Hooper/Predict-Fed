from sklearn.metrics import mean_squared_error as MSE, r2_score
from neural_network_pipeline import plot_metrics, plot_pred
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.boost import XGBoost
from predict_fed.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor

from predict_fed.plotting import rounded_scatter

from numpy.random import seed

import xgboost as xgb

seed(1)


def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.PoP_PCT_CHANGE_ANN, Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    # define a grid of the hyperparameter search space
    max_depth = [i+1 for i in range(10)]
    n_estimators = [10,100,1000]
    learning_rate = [0.1, 0.2, 0.3]
    colsample_bytree = [0.2,0.4,0.6,0.8,1]
    subsample = [0.2,0.4,0.6,0.8,1]

    params = dict(
        normalisation=True,
        smote=True,
    )

    grid = dict(
        max_depth=max_depth,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        colsample_bytree=colsample_bytree,
        subsample=subsample,
    )

    xgb_model = XGBoost(dict(max_depth=4))

    pipe = Pipeline(y=rate, features=features, model=xgb_model, split_percentages=(80, 0, 20), **params)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    searcher = GridSearchCV(estimator=xgb_model.model, scoring='neg_mean_squared_error', n_jobs=-1, cv=5, param_grid=grid, verbose=1)
    searchResults = searcher.fit(X_train, y_train)

    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("[INFO] best score is {:.2f} using {}".format(bestScore, bestParams))

    best_model = XGBoost(bestParams)

    pipe = Pipeline(y=rate, features=features, model=best_model, **params)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    y_pred, y_pred_rounded = pipe.predict(X_test)
    y_pred_valid, y_pred_rounded_valid = pipe.predict(X_valid)

    r2pred = r2_score(y_test, y_pred)
    r2rounded_pred = r2_score(y_test, y_pred_rounded)

    print("Valid MSE:", MSE(y_valid, y_pred_valid))
    print("Test MSE:", MSE(y_test, y_pred))

    plot_pred(y_pred_valid, y_pred_rounded_valid, y_valid)
    rounded_scatter(y_pred_rounded_valid, y_valid)


if __name__ == '__main__':
    main()