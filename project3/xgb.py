from xgboost import XGBClassifier
from train_pipeline import BaseTrainConfig, BaseModelAdapter
from dataclasses import dataclass
import torch


@dataclass
class XGBTrainConfig(BaseTrainConfig):
    train_val_split = 0.8


class XGBModelAdapter(BaseModelAdapter[XGBTrainConfig]):
    def train_params(self, train_batch: list):
        train_cfg = self.config.training_settings

        images, labels = train_batch
        images = self.dimred.fit_transform(images, y=labels)

        sample_size = images.shape[0]
        train_split = int(sample_size * train_cfg.train_val_split)

        # train_images = torch.tensor(images[:train_split]).to("cuda")
        # train_labels = torch.tensor(labels[:train_split]).to("cuda")
        # val_images = torch.tensor(images[train_split:]).to("cuda")
        # val_labels = torch.tensor(labels[train_split:]).to("cuda")

        train_images = torch.tensor(images[:train_split]).to("cpu")
        train_labels = torch.tensor(labels[:train_split]).to("cpu")
        val_images = torch.tensor(images[train_split:]).to("cpu")
        val_labels = torch.tensor(labels[train_split:]).to("cpu")

        self.model.fit(
            train_images,
            train_labels,
            eval_set=[(val_images, val_labels)],
            verbose=None,
        )

    def validate(self, val_batch):
        val_images, val_labels = val_batch
        # val_images = torch.tensor(self.dimred.transform(val_images)).to("cuda")
        val_images = torch.tensor(self.dimred.transform(val_images)).to("cpu")
        preds = self.model.predict(val_images)

        return preds, val_labels


class XGBoostClassifier:
    def __init__(
        self,
        in_features,
        n_estimators,
        learning_rate,
        max_depth,
        min_child_weight,
        subsample,
        colsample_bytree,
        objective,
        random_state,
        device,
    ):
        self.in_features = in_features

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective=objective,
            random_state=random_state,
            device=device,
        )

    def fit(self, x, y, eval_set: list, verbose=100):
        self.model.fit(x, y, eval_set=eval_set, verbose=verbose)

    def predict(self, val_images):
        return self.model.predict(val_images)


def main():
    print("hello")


if __name__ == "__main__":
    main()
