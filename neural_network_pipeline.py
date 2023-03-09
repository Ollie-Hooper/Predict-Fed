from matplotlib import pyplot as plt
from predict_fed.data import FedDecisions, FRED, Measure
from predict_fed.models.neural_network import NeuralNetwork
from predict_fed.pipeline import Pipeline
from sklearn import metrics
from predict_fed.plotting import rounded_scatter 
from sklearn.preprocessing import MinMaxScaler 


def main():  # This is where the script goes - the main part is just to ensure that it doesn't get run from another file
    rate = FedDecisions()

    fred_data_sources = ['GDPC1', 'PAYEMS', 'UNRATE', 'HOUST']

    features = {
        FRED(series): [Measure.YoY_PCT_CHANGE] for series in fred_data_sources
    }

    ann = NeuralNetwork(batch_size=5, epochs=40, learning_rate=0.001)

    pipe = Pipeline(y=rate, features=features, model=ann, balance=True) 



    performance, (X_train, X_valid, X_test, y_train, y_valid, y_test) = pipe.run()
    
    y_pred, y_pred_rounded, r2pred, r2rounded_pred = pipe.model.predict(X_test, y_test)
    plot_metrics(performance)
    plot_pred(y_pred, y_pred_rounded, y_test)
    y_pred_rounded = [item for sublist in y_pred_rounded for item in sublist]
    rounded_scatter(y_pred_rounded, y_test)

def plot_metrics(performance):
    history = performance[2]
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.ylabel('MSE loss')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss', pad=13)
    plt.legend(loc='upper right') 
    plt.show() 

def plot_pred(y_pred, y_pred_rounded, y_test): 
    x = [-1, 1]
    y = [-1, 1]

    #y_pred 
    plt.grid()
    plt.scatter(y_pred, y_test)
    plt.plot(x, y, linestyle='--', color='black')
    plt.ylabel("True Values")
    plt.xlabel("Predictions")
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.show()    


    #y_pred_rounded§
    







if __name__ == '__main__':
    main()
