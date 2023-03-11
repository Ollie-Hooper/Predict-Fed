from predict_fed.models.base import SKLearnModel
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


class RandomForest(SKLearnModel):
    def init_model(self):
        self.model = RandomForestRegressor(**self.params)


class XGBoost(SKLearnModel):
    def init_model(self):
        self.model = XGBRegressor(**self.params)
