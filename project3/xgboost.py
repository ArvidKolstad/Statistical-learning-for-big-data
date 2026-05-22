from xgboost import XGBClassifier


class XGBoost:
    def __init__(
        self,
    ):
        self.model = XGBClassifier()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)
