import pandas as pd
from predict_fed.data import FedDecisions, FRED, Measure
import matplotlib.pyplot as plt

rate = FedDecisions()
#rate_df = rate.get_data()
#rate_df.to_csv('rates.csv') 
#rate_df.plot()
#plt.show() 



rate = FedDecisions()
rate_df = rate.get_data()
payrolls = FRED('PAYEMS')
df = pd.DataFrame()
df['rate'] = rate_df
df['payrolls_yoy'] = payrolls.get_data(measure=Measure.YoY_PCT_CHANGE)
payrolls_series = payrolls.get_data(dates=rate_df.index, measure=Measure.YoY_PCT_CHANGE)   


df.to_csv('rates&payroll.csv') 




