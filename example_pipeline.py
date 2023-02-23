from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline


def main():
    rate = FedDecisions()

    gdp = FRED('GDPC1')
    payrolls = FRED('PAYEMS')

    feature_list = [gdp, payrolls]
    measures = [Measure.PoP_PCT_CHANGE_ANN, Measure.YoY_PCT_CHANGE]

    features = {
        feature: measures for feature in feature_list
    }

    model = NeuralNetwork(5)

    pipe = Pipeline(y=rate, features=features, model=model)

    pipe.run()


if __name__ == "__main__":
    main()
