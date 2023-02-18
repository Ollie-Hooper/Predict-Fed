import enum

import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup
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
        self.series = series.upper()
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
            formatted_df.loc[date, :] = np.flip(row.iloc[max(0, last_idx - self.freq_n):last_idx + 1].values)
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
        return df.rename(f"{self.series}_{measure.name}")


class FedDecisions(DataSource):
    def __init__(self):
        self.historical_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"
        self.recent_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        dates = self.get_meeting_dates()
        return

    def get_rate_changes(self):
        pass

    def get_meeting_dates(self):
        dates = []
        for year in range(1936, 2018):
            dates.extend(self.get_meetings_for_historical_year(year))
        dates.extend(self.get_meetings_for_recent_years())
        return dates

    def get_meetings_for_historical_year(self, year):
        url = self.historical_url + str(year) + ".htm"
        soup = self.get_soup(url)
        date_divs = soup.find_all('div', attrs={'class': 'panel-heading'})
        dates = []
        for div in date_divs:
            text = div.text.strip()
            month = text.strip().split('/')[0].split(' ')[0]
            day = text.strip().split(' ')[1].split('-')[0]
            date = pd.to_datetime(f"{month} {day} {year}")
            dates.append(date)
        return dates

    def get_meetings_for_recent_years(self):
        soup = self.get_soup(self.recent_url)
        year_panels = soup.find_all('div', attrs={'class': 'panel-default'})
        dates = []
        for year_panel in year_panels:
            year = year_panel.find('div', attrs={'class': 'panel-heading'}).text.strip().split(' ')[0]
            meetings = year_panel.find_all('div', attrs={'class': 'fomc-meeting'})
            for meeting in meetings:
                month = \
                    meeting.find('div', attrs={'class': 'fomc-meeting__month'}).text.strip().split('/')[0].split(' ')[0]
                day = meeting.find('div', attrs={'class': 'fomc-meeting__date'}).text.strip().split('-')[0].split(' ')[
                    0]
                date = pd.to_datetime(f"{month} {day} {year}")
                dates.append(date)
        return sorted(dates)

    def get_soup(self, url):
        req = requests.get(url)
        source = req.text
        return BeautifulSoup(source, 'html.parser')
