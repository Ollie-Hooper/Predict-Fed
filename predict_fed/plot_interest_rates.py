from predict_fed.data import FedDecisions, FRED
rate = FedDecisions()
rate_df = rate.get_data()
rate_df.to_csv('rates.csv')

