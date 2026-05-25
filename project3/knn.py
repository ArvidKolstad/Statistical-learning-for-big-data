from train_pipeline import BaseModelAdapter, BaseTrainConfig
from sklearn.neighbors import KNeighborsClassifier


class KNNModelAdapter(BaseModelAdapter[BaseTrainConfig]):
    def train_params(self, train_batch: list):
        train_images, train_labels = train_batch
        train_images = self.dimred.fit_transform(train_images, y=train_labels)

        self.model.fit(train_images, train_labels)

    def validate(self, val_batch):
        val_images, val_labels = val_batch
        val_images = self.dimred.transform(val_images)
        preds = self.model.predict(val_images)
        return preds, val_labels


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


def main():
    print("hello")


if __name__ == "__main__":
    main()
