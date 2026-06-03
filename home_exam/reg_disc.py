import numpy as np
from scipy.special import softmax
from dataclasses import dataclass
from numpy.linalg import inv, slogdet
from torch.utils.data import DataLoader
from train_pipeline import BaseTrainConfig, BaseModelAdapter
from data_process import B11_dataset


@dataclass
class RDATrainConfig(BaseTrainConfig):
    batch_size: int = 500


class RDAModelAdapter(BaseModelAdapter[RDATrainConfig]):
    def train_params(self, train_batch: list):
        train_cfg = self.config.training_settings
        train_images, train_labels = train_batch

        dataset = B11_dataset(train_images, train_labels)
        train_loader = DataLoader(
            dataset, batch_size=train_cfg.batch_size, num_workers=0
        )
        self.model.train(train_loader)

    def validate(self, val_batch):
        val_images, val_labels = val_batch

        preds = self.model.decision_rule(val_images)

        return preds, val_labels

    def get_probability(self, val_input):
        preds = self.model.predict_probalility(val_input)
        return preds

    def save_model(self):
        self.model.save(self.output_dir)

    def load_model(self):
        self.model.load(self.output_dir)

    def get_class(self, val_input):
        return self.model.decision_rule(val_input)


class RegularizedDiscriminantAnalysis:
    def __init__(
        self,
        in_features: int,
        classes: list,
        lmbda: float,
        gamma: float,
        load_params=None,
    ) -> None:
        self.in_features = in_features
        self.classes = classes
        self.number_of_classes = len(classes)

        self.lmbda = lmbda
        self.gamma = gamma

        self.mean_vector = np.zeros((self.number_of_classes, in_features))
        self.covariance_matrices = np.zeros(
            (self.number_of_classes, in_features, in_features)
        )
        self.inverse_covariances = np.zeros(
            (self.number_of_classes, in_features, in_features)
        )
        self.pi = np.zeros(self.number_of_classes)

        if load_params:
            self.load(load_params)

    def __call__(self, input_array) -> np.ndarray:
        return self.decision_rule(input_array)

    def save(self, save_path):
        np.savez(
            save_path,
            mean_vector=self.mean_vector,
            covariance_matrices=self.covariance_matrices,
            inverse_covariances=self.inverse_covariances,
            pi=self.pi,
        )

    def load(self, save_path):
        with np.load(save_path) as model_params:

            assert (
                self.covariance_matrices.shape
                == model_params["covariance_matrices"].shape
            )
            assert (
                self.inverse_covariances.shape
                == model_params["inverse_covariances"].shape
            )
            assert self.mean_vector.shape == model_params["mean_vector"].shape

            self.mean_vector = model_params["mean_vector"]
            self.covariance_matrices = model_params["covariance_matrices"]
            self.inverse_covariances = model_params["inverse_covariances"]
            self.pi = model_params["pi"]

    def decision_rule(self, input_array: np.ndarray) -> np.ndarray:
        x = input_array
        self.mean_vector
        _, log_det_abs = slogdet(self.covariance_matrices)

        diff = x[:, None, :] - self.mean_vector[None, :, :]

        tmp = np.einsum("bki,kij->bkj", diff, self.inverse_covariances)
        quad = (tmp * diff).sum(axis=-1)

        scores = -0.5 * log_det_abs - 0.5 * quad + np.log(self.pi)

        return np.argmax(scores, axis=1).astype(np.int64)

    def predict_probalility(self, input_array) -> np.array:
        x = input_array
        self.mean_vector
        _, log_det_abs = slogdet(self.covariance_matrices)

        diff = x[:, None, :] - self.mean_vector[None, :, :]

        tmp = np.einsum("bki,kij->bkj", diff, self.inverse_covariances)
        quad = (tmp * diff).sum(axis=-1)

        scores = -0.5 * log_det_abs - 0.5 * quad + np.log(self.pi)
        probability = softmax(scores, axis=1)
        return probability

    def validation(self, val_loader: DataLoader) -> float:
        X_all = np.concatenate([X.numpy() for X, _ in val_loader], axis=0)
        labels_all = np.concatenate([y.numpy() for _, y in val_loader], axis=0)

        pred = self.decision_rule(X_all)
        correct_classification = np.sum(labels_all == pred)
        total_classification = labels_all.shape[0]

        accuracy = correct_classification / total_classification
        return accuracy

    def train(self, train_data: DataLoader):
        n_samples = np.zeros(self.number_of_classes)
        sum_x = np.zeros((self.number_of_classes, self.in_features))
        sum_x2 = np.zeros((self.number_of_classes, self.in_features, self.in_features))

        for X_batch, labels in train_data:
            X_batch, labels = X_batch.numpy(), labels.numpy()
            for idx, k in enumerate(self.classes):
                mask = labels == k
                if not mask.any():
                    continue
                x = X_batch[mask]
                n_samples[idx] += len(x)
                sum_x[idx] += x.sum(axis=0)
                sum_x2[idx] += x.T @ x

        total_samples = np.sum(n_samples)

        self.pi = n_samples / total_samples

        self.mean_vector = sum_x / n_samples[:, None]

        S_k = (
            sum_x2 / n_samples[:, None, None]
            - self.mean_vector[:, :, None] * self.mean_vector[:, None, :]
        )

        S_pooled = np.einsum("k,kij->ij", n_samples, S_k) / total_samples

        S_reg = (1 - self.lmbda) * S_k + self.lmbda * S_pooled[None]

        if self.gamma > 0:
            avg_eig = np.einsum("kii->k", S_reg) / self.in_features
            S_reg = (1 - self.gamma) * S_reg + self.gamma * avg_eig[
                :, None, None
            ] * np.eye(self.in_features)

        self.covariance_matrices = S_reg

        try:
            self.inverse_covariances = inv(S_reg)
        except np.linalg.LinAlgError as e:
            print(f"Singular matrix encountered: {e}", flush=True)
            jitter = 1e-6 * np.eye(self.in_features)
            self.inverse_covariances = inv(S_reg + jitter)
