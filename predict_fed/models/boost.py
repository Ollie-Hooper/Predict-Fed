from predict_fed.models.base import Model
import xgboost as xgb

from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import r2_score


class XGBoost(Model):
    def __init__(self, model_params={}):
        super().__init__('XGBoost')
        self.params = model_params

    def init_model(self):
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, train_x, train_y, valid_x, valid_y):
        # Train the model
        self.trained = True
        self.init_model()
        # Fit the model
        self.model.fit(train_x, train_y)

    def evaluate(self, train_x, train_y, test_x, test_y):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")

        # Evaluate the model
        y_pred = self.model.predict(test_x)
        mse = MSE(test_y, y_pred)
        r2 = r2_score(test_y, y_pred)
        return mse, r2

    def predict(self, test_x, ):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        return self.model.predict(test_x)
