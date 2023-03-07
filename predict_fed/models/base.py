class Model:
    def __init__(self, name):
        self.name = name
        self.trained = False

    def train(self, train_x, train_y):
        self.trained = True

    def evaluate(self, test_x, test_y):
        pass

    def predict(self, test_x):
        if not self.trained:
            raise Exception(f"Model '{self.name}' has not been trained...")
        return
