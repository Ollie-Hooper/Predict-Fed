#%%
from sklearn.model_selection import train_test_split
import pandas as pd

from predict_fed.models.decision_tree import DecisionTree as dt
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.pipeline import Pipeline

# Real GDP yoy (GDPC1)  , PAYEMS ( Non Farm PayRolls) ,  UNRATE

list_of_desired_features = ['PAYEMS','GDPC1','UNRATE']


features = {
    FRED(k): [Measure.YoY_PCT_CHANGE,] for k in list_of_desired_features
}
rate = FedDecisions()


DTree = dt('squared_error')

pipe = Pipeline(y=rate, features=features, model=DTree)

performance = pipe.run()

pipe.model.visualisation()



for func in ['squared_error', 'friedman_mse', 'absolute_error']:
    for d in range(1,13):
        DTree = dt(func, d)
        pipe = Pipeline(y=rate, features=features, model=DTree)
        performance = pipe.run()


def preprocesss(data_frame):
    data= data_frame
    df= data.copy()
    removed_null= df.dropna()
    X= removed_null.iloc[:,:-1]
    y= removed_null.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test



def construct_dataframe(features_list):
    df = pd.DataFrame()

    rate = FedDecisions()

    rate_df = rate.get_data()
    features = features_list
    df['rate'] = rate_df

    for feature in features:
        fred_feature = FRED(feature)

        df[feature] = fred_feature.get_data(dates=rate_df.index, measure=Measure.YoY_PCT_CHANGE)

    return df



data_to_test = preprocesss(construct_dataframe((list_of_desired_features)))

DTree = dt('squared_error')
DTree.train(data_to_test[0],data_to_test[2])

performance = DTree.evaluate(data_to_test[0],data_to_test[2],data_to_test[1],data_to_test[3])
visualisation= DTree.visualisation()


# %%