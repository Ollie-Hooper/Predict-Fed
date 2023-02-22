from sklearn.model_selection import train_test_split
import pandas as pd
from decision_tree import DecisionTree as dt

def preprocesss(file_path):
    data= pd.read_csv(file_path)
    df= data.copy()
    df_relevant= df[['Annual_Income','Tax_Liens','Number_of_Open_Accounts','Years_of_Credit_History','Maximum_Open_Credit','Number_Credit_Problems','Bankruptcies','Current_Credit_Balance','Monthly_Debt','Credit_Score']]
    removed_null= df_relevant.dropna()
    X= removed_null.iloc[:,:-1]
    y= removed_null.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test



a= preprocesss('awsautoloantest.csv')


trial_dec_tree= dt('squared_error',a[0],a[1], a[2],a[3])

trial_dec_tree_performance= trial_dec_tree.performance()
