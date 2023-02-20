from predict_fed.models.base import Model
from predict_fed.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
import numpy as np
class DecisionTree(Model):
    def __init__(self, crit, x_train, x_test, y_train, y_test):
        super().__init__('Decision Tree')
        self.crit = crit
        self.train_x, self.train_y, self.test_x, self.test_y = x_train, y_train, x_test , y_test
        self.DTree = self.train()
        self.prediction = self.predict()

    def train(self):
        self.trained = True

        RegModel = DecisionTreeRegressor(criterion=self.crit)
        RegModel.fit(self.train_x,self.train_y)
        
        return RegModel


    def predict(self):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        
        prediction = self.DTree.predict(self.test_x)

        return prediction

    def performance(self):
        # Measuring Goodness of fit in Training data
        r2_value = r2_score(self.train_y, self.DTree.predict(self.train_x))
        print('R2 Value: ', r2_score(self.train_y, self.DTree.predict(self.train_x)))
        # Measuring accuracy on Testing Data
        accuracy_value  = 100 - (np.mean(np.abs((self.test_y - self.prediction) / self.test_y)) * 100)
        print('Accuracy', 100 - (np.mean(np.abs((self.test_y - self.prediction) / self.test_y)) * 100))
        performance_scores = [r2_value, accuracy_value]
        return performance_scores



        

    


       
    
