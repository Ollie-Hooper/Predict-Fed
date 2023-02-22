from sklearn.model_selection import train_test_split
import pandas as pd




def traintestsplit(data):
    y = data[y_col]
    X = data[features]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

