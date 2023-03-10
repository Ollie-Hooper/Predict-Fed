# %%
from predict_fed.models.base import Model
from predict_fed.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import graphviz


class DecisionTree(Model):
    def __init__(self, crit, max_depth):
        super().__init__('Decision Tree')
        self.crit = crit
        self.max_depth = max_depth
        self.model = None

    def train(self, train_x, train_y, *args):
        self.trained = True

        self.model = DecisionTreeRegressor(criterion=self.crit, max_depth=self.max_depth)
        self.model.fit(train_x, train_y)

    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")

        prediction = self.model.predict(test_x)

        return prediction

    def evaluate(self, train_x, train_y, test_x, test_y):
        predict_train_y = self.predict(train_x)
        predict_test_y = self.predict(test_x)
        r2_value = r2_score(test_y, predict_test_y)
        tree_depth = self.model.get_depth()


        # Measuring accuracy on Testing Data
        validation_mse = mean_squared_error(test_y, predict_test_y)
        training_mse = mean_squared_error(train_y, predict_train_y)

        print('Tree Depth', tree_depth, 'MSE_Validation', validation_mse, 'MSE_Train', training_mse, 'R2 Score', r2_value)

        performance_scores = [tree_depth, r2_value, validation_mse, training_mse]
        return performance_scores

    def visualisation(self):
        dot_data = export_graphviz(self.model, out_file='tree.dot')

        return dot_data

# %%