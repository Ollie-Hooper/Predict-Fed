import pandas as pd
from predict_fed.data import FedDecisions, FRED, Measure
import matplotlib.pyplot as plt

rate = FedDecisions()
rate_df = rate.get_data()
rate_df.to_csv('rates.csv')
plt.xlabel('Date', fontsize=13)
plt.ylabel('Interest rate change (%)', fontsize=13)
rate_df.plot()
plt.show()



rate = FedDecisions()
rate_df = rate.get_data()
payrolls = FRED('PAYEMS')
gdp = FRED('GDPC1')
unemployment_rate = FRED('UNRATE')

df = pd.DataFrame()
df['rate'] = rate_df
df['payrolls_yoy'] = payrolls.get_data(dates=rate_df.index, measure=Measure.YoY_PCT_CHANGE)
df['gdp_yoy'] = gdp.get_data(dates=rate_df.index, measure=Measure.YoY_PCT_CHANGE)
df['unemployment_rate_yoy'] = unemployment_rate.get_data(dates=rate_df.index, measure=Measure.YoY_CHANGE)






df.to_csv('data1.csv')











