from predict_fed.models.base import Model
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork(Model):
    def __init__(self, batch_size, epochs):
        super().__init__('Neural Network')
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, train_x, train_y):
        # assuming data is already preprocessed and normalized

        # Train the model
        self.trained = True
        num_features = train_x.shape[1]
        self.model = Sequential()
        # Add the input layer
        self.model.add(Dense(64, input_dim=num_features, activation='relu'))
        # add hidden layer
        self.model.add(Dense(12, activation='relu'))
        # add output regression layer
        self.model.add(Dense(1, activation='linear'))

        # Compile the model
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
        # Fit the model
        self.model.fit(train_x, train_y, batch_size=self.batch_size,
                       epochs=self.epochs)  # batch_size is the number of samples per gradient update for training and epochs is the number of epochs to train the model

    # both hyperparameters can be tuned to improve the model

    def evaluate(self, train_x, train_y, test_x, test_y):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")

        # Evaluate the model
        scores = self.model.evaluate(test_x, test_y)
        print(f"{self.name} Loss: {scores[0]}")
        print(f"{self.name} Accuracy: {scores[1]}")
        return scores[0], scores[1]

    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")

        pred = self.model.predict(test_x)
        rounded_pred = np.round(pred * 4) / 4
        return rounded_pred
