from sklearn.metrics import r2_score
from neural_network_pipeline import plot_metrics, plot_pred
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline 
from sklearn.model_selection import RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Dense 
import keras 
from keras.wrappers.scikit_learn import KerasRegressor

from predict_fed.plotting import rounded_scatter






def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.PoP_PCT_CHANGE_ANN, Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }


    # define a grid of the hyperparameter search space
    batch_size = [16, 32, 64]
    epochs = [15, 25, 35]
    learning_rate = [1e-3]
    hidden_layer1 = [128, 256, 512, 784]
    hidden_layer2 = [0, 128, 256, 512]
    hidden_layer3 = [0, 128, 256, 512]
    reg = [0.01, 0.001]

    grid = dict(
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        hidden_layer1=hidden_layer1,
        hidden_layer2=hidden_layer2,
        hidden_layer3=hidden_layer3,
        reg=reg
    ) 


    



    ann = NeuralNetwork(batch_size=32, epochs=1, learning_rate=0.001, hidden_layer1 = 50, hidden_layer2 = 50, hidden_layer3=40, reg = 0.3)

    pipe = Pipeline(y=rate, features=features, model=ann, bootstrap=True, smote=False) 

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()

    def get_model(batch_size=batch_size,epochs=epochs,learning_rate=learning_rate,hidden_layer1=hidden_layer1,hidden_layer2=hidden_layer2,hidden_layer3=hidden_layer3,reg=reg):#
        model = NeuralNetwork(batch_size=batch_size,epochs=epochs,learning_rate=learning_rate,hidden_layer1=hidden_layer1,hidden_layer2=hidden_layer2,hidden_layer3=hidden_layer3,reg=reg,score=False)
        model.init_model(X_train.shape[1])
        return model.model
    
    regressor = KerasRegressor(build_fn=get_model, verbose=0)

    X_train = X_train.values
    y_train = y_train.values


    searcher = RandomizedSearchCV(estimator=regressor, scoring='mse', n_jobs=-1, cv=3, param_distributions=grid)
    searchResults = searcher.fit(X_train, y_train)

    bestScore = searchResults.best_score_
    bestParams = searchResults.best_params_
    print("[INFO] best score is {:.2f} using {}".format(bestScore,bestParams))

    best_model = NeuralNetwork(**bestParams)
    best_model.init_model(X_train.shape[1])
    best_pipe = Pipeline(y=rate, features=features, model=best_model, bootstrap=False, smote=True, normalisation=True)

    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = best_pipe.run()
    
    
    y_pred, y_pred_rounded = best_pipe.predict(X_test)

    r2pred = r2_score(y_test, y_pred)
    r2rounded_pred = r2_score(y_test, y_pred_rounded)

    plot_metrics(performance)
    plot_pred(y_pred, y_pred_rounded, y_test)
    rounded_scatter(y_pred_rounded, y_test)


if __name__ == '__main__':
    main()