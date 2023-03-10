import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from keras.models import Sequential
from keras.layers import Dense 
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import keras
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler 
from keras.regularizers import l2
from sklearn.model_selection import RandomizedSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import shap
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

def build_model(batch_size, epochs, learning_rate, hidden_layer1, hidden_layer2, hidden_layer3, reg):
    num_features = train_x.shape[1]
    model = Sequential()
    # Add the hidden layer
    model.add(Dense(hidden_layer1, input_dim=num_features, activation='relu'))
    # add hidden layer
    model.add(Dense(hidden_layer2, activation='relu', kernel_regularizer=l2(reg)))
    model.add(Dense(hidden_layer3, activation='relu', kernel_regularizer=l2(reg)))
    # add output regression layer
    model.add(Dense(1, activation='linear'))


    # Configuring optimizer
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    # Compile the model
    model.compile(loss='mean_squared_error', optimizer=opt)
    # Fit the model
<<<<<<< HEAD
    #model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
=======
    model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
>>>>>>> main


    return model


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


regressor = KerasRegressor(build_fn=build_model, verbose=0)
searcher = RandomizedSearchCV(estimator=regressor, n_jobs=-1, cv=3, param_distributions=grid)
searchResults = searcher.fit(train_x, train_y) 



bestScore = searchResults.best_score_
bestParams = searchResults.best_params_
print("[INFO] best score is {:.2f} using {}".format(bestScore,bestParams)) 


model = build_model(8, 40, 0.001, 512, 256, 128, 0.3) 

# Calculate feature importance using SHAP
explainer = shap.Explainer(model.predict, train_x)
shap_values = explainer(test_x)

# Plot the SHAP summary plot
shap.summary_plot(shap_values, test_x)


# x = [i for i in range(len(test_y))]  # want this to be the dates 

# plt.plot(x, test_y, label = 'actual interest rate')
# plt.plot(x, rounded_pred, label = 'predicted interest rate')
# plt.legend()
# plt.show() 

# plt.plot(x, test_y, label = 'actual interest rate')
# plt.plot(x, rounded_pred, label = 'predicted interest rate')
# plt.legend()
# plt.show() 


# plt.scatter(test_y, rounded_pred)
# plt.xlabel("True Values")
# plt.ylabel("Predictions")
# plt.show()  

# plt.scatter(x, test_y, label = 'actual interest rate')
# plt.scatter(x, rounded_pred, label = 'predicted interest rate')
# plt.xlabel("date")
# plt.ylabel("interest rate chnge") 
# plt.legend()
# plt.show()  



