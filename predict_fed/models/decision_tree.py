#%%
from predict_fed.models.base import Model
from predict_fed.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import graphviz 

class DecisionTree(Model):
    def __init__(self, crit, x_train, x_test, y_train, y_test):
        super().__init__('Decision Tree')
        self.crit = crit
        self.train_x, self.train_y, self.test_x, self.test_y = x_train, y_train, x_test , y_test
        self.DTree = self.train()
        self.prediction = self.predict()

    def train(self):
        self.trained = True

        RegModel = DecisionTreeRegressor(criterion=self.crit, max_depth=4)
        RegModel.fit(self.train_x,self.train_y)
        
        return RegModel


    def predict(self):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        
        prediction = self.DTree.predict(self.test_x)

        return prediction

    def performance(self):
        # Measuring Goodness of fit in Training data
        r2_value = r2_score(self.test_y, self.DTree.predict(self.test_x))
        # Measuring accuracy on Testing Data
        accuracy_value = mean_squared_error(self.test_y,self.DTree.predict(self.test_x))
        training_accuracy = mean_squared_error(self.train_y,self.DTree.predict(self.train_x))

    
        print('R2 Value: ', r2_value, 'Accuracy Value _ Test (MSE)', accuracy_value, 'Accuracy Value _ Train', training_accuracy )
        
        performance_scores = [r2_value, accuracy_value]
        return performance_scores

    def visualisation(self):
        dot_data = export_graphviz(self.DTree, out_file='tree.dot')

        return dot_data


        

    


       
    

# %%
