from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.decision_tree import DecisionTree
from predict_fed.pipeline import Pipeline


def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    decision_tree = DecisionTree('squared_error')

    pipe = Pipeline(y=rate, features=features, model=decision_tree)

    performance = pipe.run()

    pipe.model.visualisation()


if __name__ == '__main__':
    main()
