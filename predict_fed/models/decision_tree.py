from predict_fed.models.base import Model
from predict_fed.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

class DecisionTree(Model):
    def __init__(self, crit):
        super().__init__('Decision Tree')
        self.crit=crit

    def train(self, train_x, train_y):
        self.trained = True

        RegModel = DecisionTreeRegressor(criterion=crit)
        DTree= RegModel.fit(X_train,y_train)
        
        return DTree


    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        
        prediction= train.predict(X_test)

        return prediction




        

    
        #Measuring Goodness of fit in Training data
        from sklearn.metrics import r2_score
        print('R2 Value:', r2_score(y_train, DTree.predict(X_train)))
        #Measuring accuracy on Testing Data
        print('Accuracy',100- (np.mean(np.abs((y_test - prediction) / y_test)) * 100))

       
    
