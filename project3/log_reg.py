from train_pipeline import BaseTrainConfig, BaseModelAdapter
from sklearn.linear_model import LogisticRegression


class LogRegAdapter(BaseModelAdapter[BaseTrainConfig]):
    def train_params(self, train_batch: list):
        train_images, train_labels = train_batch
        train_images = self.dimred.fit_transform(train_images, y=train_labels)
        self.model.train(train_images, train_labels)

    def validate(self, val_batch):
        val_images, val_labels = val_batch
        val_images = self.dimred.transform(val_images)

        preds = self.model.predict(val_images)
        return preds, val_labels


class LogisticRegressionModel:
    def __init__(self, in_features: int):
        self.infeatures = in_features
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
        )

    def train(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, Y):
        return self.model.score(X, Y)
