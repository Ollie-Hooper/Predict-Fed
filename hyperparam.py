import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.kers import Sequential



# importing data

df = pd.read_csv('data1 copy.csv') 

df1 = df.drop([i for i in range(151)])  




# Separate Target Variable and Predictor Variables
TargetVariable=['rate']
Predictors=['payrolls_yoy', 'gdp_yoy', 'unemployment_rate_yoy']
 
X=df1[Predictors].values
y=df1[TargetVariable].values



train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42)  


epoch_paramspace = [num for num in range(100, 501, 25)] 

batch_paramspace = [num for num in range(1, 16)]

accuracy_matrix = np.ones((len(epoch_paramspace),len(batch_paramspace)))



for i in range(len(epoch_paramspace)) :
    for j in range(len(batch_paramspace)):
        num_features = train_x.shape[1]
        model = Sequential()
        # Add the input layer
        model.add(Dense(64, input_dim=num_features, activation='relu'))
        #add hidden layer
        model.add(Dense(12, activation='relu'))
        #add output regression layer
        model.add(Dense(1, activation='linear')) 

        # Compile the model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 
        # Fit the model
        model.fit(train_x, train_y, batch_size=batch_paramspace[j], epochs=epoch_paramspace[i]) 
        
        scores = model.evaluate(test_x, test_y) 
        
        accuracy_matrix[i][j] = scores[1] 