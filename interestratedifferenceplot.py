from predict_fed.data import FedDecisions, FRED 
import matplotlib.pyplot as plt

rate = FedDecisions()
rate_df = rate.get_data()
rate_df.to_csv('rates.csv') 
rate_df.plot()
plt.show()
