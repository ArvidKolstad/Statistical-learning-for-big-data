from train_pipeline import BaseModelAdapter, BaseTrainConfig
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class KNNModelAdapter(BaseModelAdapter[BaseTrainConfig]):
    def train_params(self, train_batch: list):
        train_images, train_labels = train_batch
        self.model.fit(train_images, train_labels)

    def validate(self, val_batch):
        val_images, val_labels = val_batch
        preds = self.model.predict(val_images)
        return preds, val_labels

    def get_probability(self, val_input) -> np.ndarray:
        probability = self.model.get_probability(val_input)
        return probability


class KNNClassifier:
    def __init__(
        self,
        in_features,
        n_neighbors,
        weights,
        algorithm,
        leaf_size,
        p,
        metric,
        n_jobs,
    ):
        self.in_features = in_features

        self.model = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            metric=metric,
            n_jobs=n_jobs,
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def get_probability(self, x):
        return self.model.predict_proba(x)


def main():
    print("hello")


if __name__ == "__main__":
    main()
