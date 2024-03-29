from predict_fed.models.base import Model
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler 
from keras.regularizers import l2



class NeuralNetwork(Model):
    def __init__(self, batch_size, epochs, learning_rate, hidden_layer1, hidden_layer2, hidden_layer3, reg, score=True):
        super().__init__('Neural Network')
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.hidden_layer3 = hidden_layer3
        self.reg = reg
        self.score = score

    def init_model(self, num_features):
        self.model = Sequential()
        # Add the hidden layer
        self.model.add(Dense(self.hidden_layer1, input_dim=num_features, activation='relu'))
        # add hidden layer
        self.model.add(Dense(self.hidden_layer2, activation='relu', kernel_regularizer=l2(self.reg)))
        self.model.add(Dense(self.hidden_layer3, activation='relu', kernel_regularizer=l2(self.reg)))
        # add output regression layer
        self.model.add(Dense(1, activation='linear'))


        # Configuring optimizer
        opt = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Compile the model
        if self.score:
            self.model.compile(loss='mean_squared_error', optimizer=opt, metrics=r2_score, run_eagerly=True)
        else:
            self.model.compile(loss='mean_squared_error', optimizer=opt)
    
    def train(self, train_x, train_y, valid_x, valid_y):
        # assuming data is already preprocessed and normalized
        #min_max_scaler = MinMaxScaler()
        #train_x = min_max_scaler.fit_transform(train_x)
        #valid_x = min_max_scaler.fit_transform(valid_x)


        # Train the model
        self.trained = True
        num_features = train_x.shape[1]
        self.init_model(num_features)
        # Fit the model
        self.history = self.model.fit(train_x, train_y, batch_size=self.batch_size,
                       epochs=self.epochs, validation_data=(valid_x, valid_y))  # batch_size is the number of samples per gradient update for training and epochs is the number of epochs to train the model


        
        
    
    
    # both hyperparameters can be tuned to improve the model

    def evaluate(self, train_x, train_y, test_x, test_y):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")

        #min_max_scaler = MinMaxScaler()
        #test_x = min_max_scaler.transform(test_x)
        
        # Evaluate the model
        scores = self.model.evaluate(test_x, test_y)
        print(f"{self.name} Loss: {scores[0]}")
        print(f"{self.name} R2: {scores[1]}") 
        
        return scores[0], scores[1], self.history

    def predict(self, test_x,):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        return self.model.predict(test_x)
