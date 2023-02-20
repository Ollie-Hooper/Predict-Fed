from predict_fed.models.base import Model


class DecisionTree(Model):
    def __init__(self):
        super().__init__('Decision Tree')

    def train(self, train_x, train_y):
        self.trained = True

    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        return
