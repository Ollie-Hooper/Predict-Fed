import pandas as pd

class DataSource:
    def __init__(self, name):
        self.name = name

    def get_data(self) -> pd.DataFrame:
        return