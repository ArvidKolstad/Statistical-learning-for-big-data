from reg_disc import RegularizedDiscriminantAnalysis
from knn import KNNClassifier
from torch.utils.data import DataLoader
from data_process import B11_dataset
import numpy as np
from train_pipeline import BaseModelAdapter, BaseTrainConfig
from dataclasses import dataclass


class HierarchyModelAdapter(BaseModelAdapter[BaseTrainConfig]):
    def train_params(self, train_batch: list):
        train_inputs, true_labels = train_batch

        self.model.train(train_inputs, true_labels)

    def validate(self, val_batch):
        val_images, true_labels = val_batch
        preds = self.model.validate(val_images)
        return preds, true_labels


class HierarchyModel:
    def __init__(
        self,
        in_features,
        classes_minority,
        lmbda,
        gamma,
        n_neighbors,
        weights,
        algorithm,
        leaf_size,
        p,
        metric,
        n_jobs,
    ):

        self.majority_model = KNNClassifier(
            in_features, n_neighbors, weights, algorithm, leaf_size, p, metric, n_jobs
        )
        self.minority_model = RegularizedDiscriminantAnalysis(
            in_features,
            classes_minority,
            lmbda,
            gamma,
        )

    def train(self, inputs: np.ndarray, labels: np.ndarray):
        minority_mask = labels != 0

        extra_labels = labels.copy()

        extra_labels[minority_mask] = 1
        self.majority_model.fit(inputs, extra_labels)

        minority_labels = labels[minority_mask] - 1
        minority_inputs = inputs[minority_mask]

        minority_dataset = B11_dataset(minority_inputs, minority_labels)
        minority_dataloader = DataLoader(
            minority_dataset,
            batch_size=500,
            shuffle=False,
            num_workers=0,
        )
        self.minority_model.train(minority_dataloader)

    def validate(self, inputs):
        preds = self.majority_model.predict(inputs).astype(int)
        minority_mask = preds == 1
        minority_inputs = inputs[minority_mask]
        minority_labels = minority_class_mapping(
            self.minority_model.decision_rule(minority_inputs)
        )

        preds[minority_mask] = minority_labels
        return preds


def minority_class_mapping(labels) -> np.ndarray:
    mapping_dict = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6}
    minority_class_mapping = np.array([mapping_dict[i] for i in range(6)])
    return minority_class_mapping[labels]
