import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#from tensorflow import keras
#from tensorflow.keras import layers
#from tensorflow.kers import Sequential



# importing data

df = pd.read_csv('data1.csv') 

df1 = df.drop([i for i in range(151)])  




# Separate Target Variable and Predictor Variables
TargetVariable=['rate']
Predictors=['payrolls_yoy', 'gdp_yoy', 'unemployment_rate_yoy']
 
X=df1[Predictors].values
y=df1[TargetVariable].values



train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42) 


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
model.fit(train_x, train_y, batch_size=5, epochs=100) 


    # Configuring optimizer
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=opt)
    # Fit the model
    #model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)

pred  = model.predict(test_x) 
rounded_pred = np.round(pred* 4) / 4 





x = [i for i in range(len(test_y))]  # want this to be the dates 

plt.plot(x, test_y, label = 'actual interest rate')
plt.plot(x, rounded_pred, label = 'predicted interest rate')
plt.legend()
plt.show() 

plt.plot(x, test_y, label = 'actual interest rate')
plt.plot(x, rounded_pred, label = 'predicted interest rate')
plt.legend()
plt.show() 


plt.scatter(test_y, rounded_pred)
plt.xlabel("True Values")
plt.ylabel("Predictions")
plt.show()  

plt.scatter(x, test_y, label = 'actual interest rate')
plt.scatter(x, rounded_pred, label = 'predicted interest rate')
plt.xlabel("date")
plt.ylabel("interest rate chnge") 
plt.legend()
plt.show()  



