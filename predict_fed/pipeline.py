import os

import pandas as pd

from sklearn.model_selection import train_test_split

from predict_fed.data import DataSource


class Pipeline:
    def __init__(self, y, features, model, evaluator=None):
        self.y = y
        self.feature_sources = features
        self.model = model
        self.evaluator = evaluator
        self.y_col = None
        self.features = []

    @staticmethod
    def get_cached_df(source):
        df_path = f'data_cache/{source.name}.csv'
        if os.path.exists(df_path):
            df = pd.read_csv(df_path, index_col=0, parse_dates=True)
        else:
            df = source.get_data()
            if not os.path.exists('data_cache/'):
                os.mkdir('data_cache')
            df.to_csv(df_path)
        return df

    def run(self):
        data = pd.DataFrame()
        y = self.get_cached_df(self.y).iloc[:,0]
        self.y_col = y.name
        data[y.name] = y
        for feature, measures in self.feature_sources.items():
            df = self.get_cached_df(feature)
            df = DataSource.known_on_date(df, data.index)
            for measure in measures:
                measure_series = feature.apply_measure(df, measure)
                data[measure_series.name] = measure_series
                self.features.append(measure_series.name)
        X_train, X_test, y_train, y_test = self.split_data(data)
        self.model.train(X_train, y_train)

    def split_data(self, data):
        y = data[self.y_col]
        X = data[self.features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
