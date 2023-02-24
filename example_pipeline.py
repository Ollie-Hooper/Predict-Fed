from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline


def main():
    rate = FedDecisions()

    gdp = FRED('GDPC1')
    payrolls = FRED('PAYEMS')
    unemployment_rate = FRED('UNRATE')
    consumer_sentiment = FRED('UMCSENT')

    pct_change_feature_list = [gdp, payrolls]  # We want the annualised percentage changes PoP and YoY for these
    pct_change_measures = [Measure.PoP_PCT_CHANGE_ANN, Measure.YoY_PCT_CHANGE]

    change_feature_list = [unemployment_rate]  # But we only care about absolute PoP/YoY change in these (maybe)
    change_measures = [Measure.PoP_CHANGE, Measure.YoY_CHANGE]

    #  Features is a dictionary with the data source as the key and a list of measures as its value
    features = {
        **{
            feature: pct_change_measures for feature in pct_change_feature_list
        },
        **{
            feature: change_measures for feature in change_feature_list
        },
        consumer_sentiment: [Measure.VALUE],  # Maybe we don't care about the change but what the current level is
    }

    model = NeuralNetwork(5, 1000)

    pipe = Pipeline(y=rate, features=features, model=model)

    pipe.run()


if __name__ == "__main__":
    main()
