import enum

import numpy as np
import pandas as pd
from fredapi import Fred


class Measure(enum.Enum):
    PoP_CHANGE = 1
    YoY_CHANGE = 2
    PoP_PCT_CHANGE = 3
    PoP_PCT_CHANGE_ANN = 4
    YoY_PCT_CHANGE = 5


class DataSource:
    def __init__(self, name):
        self.name = name

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()

    @staticmethod
    def known_on_date(df, dates):
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        new_df = pd.DataFrame(index=dates, columns=df.columns)
        for date in dates:
            new_df.loc[date] = df.loc[:date].iloc[-1]
        return new_df

    @staticmethod
    def change(df, col1, col2):
        #  col2-col1
        return df[col2] - df[col1]

    @staticmethod
    def pct_change(df, col1, col2):
        # % change col1->col2
        return DataSource.change(df, col1, col2) / df[col1]

    @staticmethod
    def ann_pct_change(df, col1, col2, periods_in_year, compounding=False):
        if compounding:
            return (1 + DataSource.pct_change(df, col1, col2)) ** periods_in_year - 1
        else:
            return DataSource.pct_change(df, col1, col2) * periods_in_year


class FRED:
    def __init__(self, series, start_date=None, end_date=None, is_compounding=False, api_key=None):
        self.series = series
        self.start_date = start_date
        self.end_date = end_date
        if not api_key:
            with open('fred_api_key.cfg', 'r') as f:
                self.api_key = f.read()
        else:
            self.api_key = api_key
        self.fred = Fred(api_key=self.api_key)
        self.freq = None
        self.freq_n = None
        self.freq_map = {
            'M': 12,
            'Q': 4,
            'Y': 1,
        }
        self.get_info()
        self.is_compounding = is_compounding

    def get_info(self):
        info = self.fred.get_series_info(self.series)
        self.freq = info['frequency_short']
        self.freq_n = self.freq_map[self.freq]

    def get_data(self, dates=None, measure=None):
        raw_df = self.get_raw_data()
        df = self.format_data(raw_df)
        if dates:
            df = DataSource.known_on_date(df, dates)
        if measure:
            df = self.apply_measure(df, measure)
        return df

    def get_raw_data(self):
        return self.fred.get_series_all_releases(self.series)

    def format_data(self, df):
        n_columns = self.freq_n + 1
        df = df.set_index(['realtime_start', 'date'])['value'].unstack('date').ffill(axis=0)
        formatted_df = pd.DataFrame(index=df.index, columns=list(range(n_columns)))
        for date, row in df.iterrows():
            last_idx = row.index.get_loc(row.last_valid_index())
            formatted_df.loc[date, :] = np.flip(row.iloc[max(0, last_idx - n_columns):last_idx].values)
        return formatted_df

    def apply_measure(self, df, measure):
        match measure:
            case Measure.PoP_CHANGE:
                df = DataSource.change(df, 1, 0)
            case Measure.YoY_CHANGE:
                df = DataSource.change(df, self.freq_n, 0)
            case Measure.PoP_PCT_CHANGE:
                df = DataSource.pct_change(df, 1, 0)
            case Measure.PoP_PCT_CHANGE_ANN:
                df = DataSource.ann_pct_change(df, 1, 0, self.freq_n, self.is_compounding)
            case Measure.YoY_PCT_CHANGE:
                df = DataSource.pct_change(df, self.freq_n, 0)
        return df
