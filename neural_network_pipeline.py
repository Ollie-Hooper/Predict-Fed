from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline


def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    ann = NeuralNetwork(batch_size=5, epochs=100)

    pipe = Pipeline(y=rate, features=features, model=ann)

    performance = pipe.run()


if __name__ == '__main__':
    main()
