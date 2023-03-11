from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error as MSE

from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.boost import XGBoost
from predict_fed.pipeline import Pipeline
from sklearn import metrics
from predict_fed.plotting import rounded_scatter, plot_pred
from sklearn.preprocessing import MinMaxScaler


def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    bestParams = {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 10, 'n_estimators': 1000,
                  'subsample': 0.4}

    boost = XGBoost(bestParams)

    pipe = Pipeline(y=rate, features=features, model=boost, smote=True, normalisation=True)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    y_pred, y_pred_rounded = pipe.predict(X_test)
    y_pred_valid, y_pred_rounded_valid = pipe.predict(X_valid)

    r2pred = r2_score(y_test, y_pred)
    r2rounded_pred = r2_score(y_test, y_pred_rounded)

    print("Valid MSE:", MSE(y_valid, y_pred_valid))
    print("Test MSE:", MSE(y_test, y_pred))

    print(performance)

    #plot_metrics(performance)
    plot_pred(y_pred, y_pred_rounded, y_test)
    rounded_scatter(y_pred_rounded, y_test)


if __name__ == '__main__':
    main()
