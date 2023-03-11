import math
import os

import numpy as np
import pandas as pd
import smogn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

from predict_fed.data import DataSource


class Pipeline:
    def __init__(self, y, features, model=None, test=False, split_percentages=(60, 20, 20), balance=False, bootstrap=False,
                 bootstrap_samples=1000, normalisation=False, cross_valid=False, n_chunks=5, chunk_n=0,
                 infer_dates=False, smote=False):
        self.y = y
        self.feature_sources = features
        self.model = model
        self.test = test
        self.split_percentages = split_percentages
        self.balance = balance
        self.bootstrap = bootstrap
        self.bootstrap_samples = bootstrap_samples
        self.normalisation = normalisation
        self.cross_valid = cross_valid
        self.n_chunks = n_chunks
        self.chunk_n = chunk_n
        self.infer_dates = infer_dates
        self.smote = smote
        self.min_max_scaler = MinMaxScaler()
        self.y_col = None
        self.features = []

    def run(self):
        data = self.get_dataframe()

        X_train, X_valid, X_test, y_train, y_valid, y_test = self.split_data(data)

        if not self.model:
            return X_train, X_valid, X_test, y_train, y_valid, y_test

        print(f"Size of training set: {len(y_train)}")
        if X_valid is not None:
            print(f"Size of validation set: {len(y_valid)}")
        print(f"Size of testing set: {len(y_test)}")

        self.model.train(X_train, y_train, X_valid, y_valid)

        data = (X_train, X_valid, X_test, y_train, y_valid, y_test)
        if self.test or X_valid is None:
            return self.model.evaluate(X_train, y_train, X_test, y_test), data
        else:
            return self.model.evaluate(X_train, y_train, X_valid, y_valid), data

    def predict(self, X_test):
        pred = self.model.predict(X_test).flatten()
        rounded_pred = np.round(pred * 4) / 4
        return pred, rounded_pred

    def get_dataframe(self):
        data = pd.DataFrame()
        y = self.get_cached_df(self.y)
        self.y_col = y.name
        data[y.name] = y
        for feature, measures in self.feature_sources.items():
            df = self.get_cached_df(feature)
            df = DataSource.known_on_date(df, data.index)
            for measure in measures:
                measure_series = feature.apply_measure(df, measure)
                data[measure_series.name] = measure_series
                self.features.append(measure_series.name)
        for col in data:
            data[col] = data[col].astype(np.float64)
        before = len(data)
        data = data.dropna()
        after = len(data)
        print(f'Lost {before - after} out of {before} data points by removing nans.')
        return data

    def get_cached_df(self, source):
        df_path = f'data_cache/{source.name}_infer.csv' if self.infer_dates else f'data_cache/{source.name}.csv'
        if os.path.exists(df_path):
            df = pd.read_csv(df_path, index_col=0, parse_dates=True)
            if len(df.columns) == 1:
                df = df[df.columns[0]]
        else:
            df = source.get_data(infer_dates=self.infer_dates)
            if not os.path.exists('data_cache/'):
                os.mkdir('data_cache')
            df.to_csv(df_path)
        return df

    def split_data(self, data):
        y = data[self.y_col]
        X = data[self.features]
        train_size = self.split_percentages[0] / sum(self.split_percentages)
        valid_size = self.split_percentages[1] / sum(self.split_percentages)
        test_size = self.split_percentages[2] / sum(self.split_percentages)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - train_size, random_state=1)
        if self.balance:
            X_train, y_train = self.balance_data(X_train, y_train)
        if self.bootstrap:
            X_train, y_train = self.bootstrap_data(X_train, y_train)
        if self.smote:
            X_train, y_train = self.smote_data(X_train, y_train)
        if not self.cross_valid:
            if self.split_percentages[1] == 0:
                X_valid, y_valid = None, None
            else:
                X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                                    test_size=test_size / (valid_size + test_size),
                                                                    random_state=1)
        else:
            X_train, X_valid, y_train, y_valid = self.get_cross_valid(X_train, y_train)
        if self.normalisation:
            X_train, X_valid, X_test = self.normalise_data(X_train, X_valid, X_test)
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def smote_data(self, X_train, y_train):
        rg = [
            [min(y_train), 1, 0],
            [0, 0, 0],
            [max(y_train), 1, 0],
        ]

        train = X_train.copy()
        train['y'] = y_train
        train = smogn.smoter(train.reset_index(drop=True), samp_method='extreme', y='y', rel_thres=0.1,
                             rel_method='manual', rel_ctrl_pts_rg=rg)
        y_train = train['y']
        X_train = train[[c for c in train if c != 'y']]
        return X_train, y_train

    def get_cross_valid(self, X_train, y_train):
        i = self.chunk_n
        chunk_size = math.ceil(len(y_train) / self.n_chunks)
        start = i * chunk_size
        end = (i + 1) * chunk_size
        X_valid = X_train.iloc[start:end]
        y_valid = y_train.iloc[start:end]
        X_train = pd.concat([X_train.iloc[:start], X_train.iloc[end:]])
        y_train = pd.concat([y_train.iloc[:start], y_train.iloc[end:]])
        return X_train, X_valid, y_train, y_valid

    def balance_data(self, X_train, y_train):
        before = len(y_train)
        no_change = y_train[y_train == 0].index
        changes = len(y_train) - len(no_change)
        X_train = X_train.drop(no_change[:len(no_change) - changes])
        y_train = y_train.drop(no_change[:len(no_change) - changes])
        after = len(y_train)
        print(f"Lost {before - after} out of {before} data points by balancing the training set.")
        return X_train, y_train

    def bootstrap_data(self, X_train, y_train):
        train = X_train.copy()
        train['y'] = y_train
        train = resample(train, n_samples=self.bootstrap_samples)
        y_train = train['y']
        X_train = train[[c for c in train.columns if c != 'y']]
        return X_train, y_train

    def normalise_data(self, X_train, X_valid, X_test):
        X_train = self.min_max_scaler.fit_transform(X_train)
        if X_valid is not None:
            X_valid = self.min_max_scaler.transform(X_valid)
        X_test = self.min_max_scaler.transform(X_test)
        return X_train, X_valid, X_test
