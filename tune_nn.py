from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline 
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense 
import keras 
from keras.wrappers.scikit_learn import KerasRegressor






def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }


    # define a grid of the hyperparameter search space
    batch_size = [4, 8, 16, 32, 64]
    epochs = [40, 80, 120]
    learning_rate = [1e-2, 1e-3, 1e-4]
    hidden_layer1 = [256, 512, 784]
    hidden_layer2 = [0, 128, 256, 512]
    hidden_layer3 = [0, 128, 256, 512]
    reg = [0.3, 0.4, 0.5]

    grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        hidden_layer3=hidden_layer3,
        reg=reg
    ) 


    



    ann = NeuralNetwork(batch_size=32, epochs=120, learning_rate=0.001, hidden_layer1 = 50, hidden_layer2 = 50, hidden_layer3=40, reg = 0.3)

    pipe = Pipeline(y=rate, features=features, model=ann, balance=True) 



    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()
    regressor = KerasRegressor(build_fn=pipe.model, verbose=0)

    searcher = RandomizedSearchCV(estimator=pipe.model, n_jobs=-1, cv=3, param_distributions=grid)
    searchResults = searcher.fit(X_train, y_train)

    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("[INFO] best score is {:.2f} using {}".format(bestScore,bestParams))

if __name__ == '__main__':
    main()