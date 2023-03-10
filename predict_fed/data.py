import datetime
import enum

import numpy as np
import pandas as pd
import requests

from bs4 import BeautifulSoup
from fredapi import Fred


class Measure(enum.Enum):
    VALUE = 1
    PoP_CHANGE = 2
    YoY_CHANGE = 3
    PoP_PCT_CHANGE = 4
    PoP_PCT_CHANGE_ANN = 5
    YoY_PCT_CHANGE = 6


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
        dates = pd.DataFrame(index=dates).loc[df.index[0]:].index
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

    @staticmethod
    def apply_measure(df, measure, name, col1=None, col2=None, year_col=None, freq=None, compounding=None):
        match measure:
            case Measure.VALUE:
                df = df.iloc[:, 0]
                return df.rename(name)
            case Measure.PoP_CHANGE:
                df = DataSource.change(df, col1, col2)
            case Measure.YoY_CHANGE:
                df = DataSource.change(df, year_col, col1)
            case Measure.PoP_PCT_CHANGE:
                df = DataSource.pct_change(df, col1, col2)
            case Measure.PoP_PCT_CHANGE_ANN:
                df = DataSource.ann_pct_change(df, col1, col2, freq, compounding)
            case Measure.YoY_PCT_CHANGE:
                df = DataSource.pct_change(df, year_col, col1)
        return df.rename(f"{name}_{measure.name}")


class FRED(DataSource):
    def __init__(self, series, start_date=None, end_date=None, is_compounding=False, api_key=None):
        super().__init__(series)
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
            'D': 365,
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

    def get_data(self, raw=False, latest=False, dates=None, measure=None, infer_dates=False):
        raw_df = self.get_raw_data(latest)
        if raw:
            return raw_df
        df = self.format_vintage_data(raw_df, infer_dates)
        if dates is not None:
            df = DataSource.known_on_date(df, dates)
        if measure:
            df = self.apply_measure(df, measure)
        return df

    def get_raw_data(self, latest):
        return self.fred.get_series_all_releases(self.series) if not latest else self.fred.get_series_all_releases(
            self.series, datetime.date.today() - datetime.timedelta(days=2))

    def format_vintage_data(self, df, infer_dates):
        n_columns = self.freq_n + 1
        if infer_dates:
            df = self.infer_publishing_dates(df)
        df = df.set_index(['realtime_start', 'date'])['value'].unstack('date').ffill(axis=0)
        formatted_df = pd.DataFrame(index=df.index, columns=[str(i) for i in range(n_columns)])
        for date, row in df.iterrows():
            last_idx = row.index.get_loc(row.last_valid_index())
            new_row = np.flip(row.iloc[max(0, last_idx - self.freq_n):last_idx + 1].values)
            formatted_df.loc[date, :] = np.pad(new_row, (0, n_columns - len(new_row)), 'constant',
                                               constant_values=np.nan)
        return formatted_df

    def infer_publishing_lag(self, df):
        start = df.realtime_start.iloc[0]
        dff = df.copy()
        dff['td'] = df.realtime_start - df.date
        return dff.groupby('date').min().loc[start:].td.mean().components.days

    def infer_publishing_dates(self, df):
        delta = datetime.timedelta(self.infer_publishing_lag(df))
        first_row = df.set_index(['realtime_start', 'date'])['value'].unstack('date').iloc[0, :].rename('value')
        new_df = pd.DataFrame(first_row).dropna()
        new_df['realtime_start'] = new_df.index + delta
        new_df = new_df.reset_index()
        df = pd.concat([new_df, df])
        return df

    def apply_measure(self, df, measure, *args, **kwargs):
        return super().apply_measure(df, measure, self.series, '1', '0', str(self.freq_n), self.freq_n,
                                     self.is_compounding)


class FedDecisions(DataSource):
    def __init__(self):
        super().__init__('FedRateChanges')
        self.historical_url = "https://www.federalreserve.gov/monetarypolicy/fomchistorical"
        self.recent_url = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def get_data(self, *args, **kwargs) -> pd.DataFrame:
        thresh = datetime.timedelta(3)
        offset = datetime.timedelta(2)
        meeting_dates = self.get_meeting_dates()
        target_rate = self.get_target_rate()
        target_rate_diff = target_rate.diff()
        rate_change_dates = target_rate.index
        decision_dates = [*rate_change_dates]
        for m_date in reversed(meeting_dates):
            likely_change = False
            for c_date in reversed(rate_change_dates):
                if m_date - thresh <= c_date <= m_date + thresh:
                    likely_change = True
                    break
            if not likely_change:
                decision_dates.append(m_date + offset)
        target_rate_diff = target_rate_diff.reindex(sorted(decision_dates)).fillna(0).rename('rate')
        return target_rate_diff.loc[rate_change_dates[0]:datetime.date.today()]

    def get_target_rate(self):
        disc_target = FRED("DFEDTAR")
        disc_data = disc_target.get_data(raw=True)
        target = FRED("DFEDTARL")
        data = target.get_data(raw=True, latest=True)
        target_rate = pd.concat([disc_data, data]).set_index('date')['value']
        target_rate = target_rate[target_rate.diff() != 0]
        return target_rate

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
