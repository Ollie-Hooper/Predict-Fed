from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, data_sources, model, evaluator):
        self.data_sources = data_sources
        self.model = model
        self.evaluator = evaluator
        self.features = []

    def run(self):
        data = pd.DataFrame()
        for source in self.data_sources:
            df = source.get_data()
            data.append(df)
        X_train, X_test, y_train, y_test = self.split_data(data)
        self.model.train(X_train, y_train)

    def split_data(self, data):
        y = data['rate']
        X = data[self.features]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

