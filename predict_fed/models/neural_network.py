from predict_fed.models.base import Model 
from keras.models import Sequential
from keras.layers import Dense



class NeuralNetwork(Model):
    def __init__(self):
        super().__init__('Neural Network')

    def train(self, train_x, train_y):
       #assuming data is already preprocessed and normalized
       
        # Train the model
        self.trained = True 
        num_features = train_x.shape[1]
        Model = Sequential()
        # Add the input layer
        Model.add(Dense(12, input_dim=num_features, activation='relu'))
        #add hidden layer
        Model.add(Dense(8, activation='relu'))
        #add output regression layer
        Model.add(Dense(1, activation='relu')) 

        # Compile the model
        Model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['accuracy']) 
        # Fit the model
        Model.fit(train_x, train_y, batch_size=5, epochs=150)   #batch_size is the number of samples per gradient update for training and epochs is the number of epochs to train the model
        #both hyperparameters can be tuned to improve the model
        
        return Model





    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        
        pred  = self.Model.predict(test_x)
        return pred  
    
