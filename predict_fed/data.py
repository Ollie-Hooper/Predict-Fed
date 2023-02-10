import numpy as np
import pandas as pd
from fredapi import Fred


class DataSource:
    def __init__(self, name):
        self.name = name

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame()


class FRED:
    def __init__(self, series, start_date=None, end_date=None, api_key=None):
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
        self.get_info()
        self.freq_map = {
            'M': 13,
            'Q': 5,
            'Y': 2,
        }

    def get_info(self):
        info = self.fred.get_series_info(self.series)
        self.freq = info['frequency_short']

    def get_data(self):
        raw_df = self.get_raw_data()
        df = self.format_data(raw_df)
        return df

    def get_raw_data(self):
        return self.fred.get_series_all_releases(self.series)

    def format_data(self, df):
        n_columns = self.freq_map[self.freq]
        df = df.set_index(['realtime_start', 'date'])['value'].unstack('date').ffill(axis=0)
        formatted_df = pd.DataFrame(index=df.index, columns=list(range(n_columns)))
        for date, row in df.iterrows():
            last_idx = row.index.get_loc(row.last_valid_index())
            formatted_df.loc[date, :] = np.flip(row.iloc[max(0, last_idx - n_columns):last_idx].values)
        return formatted_df
