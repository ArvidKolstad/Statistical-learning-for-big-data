from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv, det


class AnimalPictures(Dataset):
    def __init__(self, in_features, labels):
        self.in_features = in_features
        self.labels = labels

    def __len__(self):
        return self.in_features.shape[0]

    def __getitem__(self, idx):
        x = self.in_features[idx]
        y = self.labels[idx]

        return x, y


class RegularizedDiscriminantAnalysis:
    def __init__(
        self, in_features: int, classes: int, lmbda: float, gamma: float
    ) -> None:
        self.in_features = in_features
        self.classes = classes

        self.covariance_matrix = np.identity(in_features)
        self.inverse_covariance = np.identity(in_features)
        self.mean_vector = np.zeros((in_features, classes))
        self.pi = np.zeros(classes)

        self.lmbda = lmbda
        self.gamma = gamma

    def __call__(self, input_array) -> int:
        return self.decision_rule(input_array)

    def decision_rule(self, input_array: np.ndarray) -> int:
        x = input_array

        scores = []

        log_det = np.log(np.linalg.det(self.covariance_matrix))

        for k in range(self.classes):
            diff = x - self.mean_vector[:, k]
            quad_term = diff.T @ self.inverse_covariance @ diff

            score = -0.5 * log_det - 0.5 * quad_term + np.log(self.pi[k])
            scores.append(score)

        return int(np.argmax(scores))

    def validation(self, val_loader: DataLoader) -> float:
        correct_classification = 0
        total_classification = 0
        for X_batch, labels in val_loader:
            X_batch, labels = X_batch.numpy(), labels.numpy()
            for idx, in_features in enumerate(X_batch):
                total_classification += 1
                pred = self.decision_rule(in_features)
                if pred == int(labels[idx]):
                    correct_classification += 1
        accuracy = correct_classification / total_classification
        return accuracy

    def train(
        self,
        train_data: DataLoader,
        val_data: DataLoader,
    ):
        n_samples = np.zeros(self.classes)
        sum_x = np.zeros((self.classes, self.in_features))
        sum_x2 = np.zeros((self.classes, self.in_features, self.in_features))

        for X_batch, labels in train_data:
            print(np.sum(n_samples))
            X_batch, labels = X_batch.numpy(), labels.numpy()
            for label in range(self.classes):
                mask = labels == label
                if not np.any(mask):
                    continue

                x = X_batch[mask]
                n_samples[label] += x.shape[0]
                sum_x[label] += x.sum(axis=0)
                sum_x2[label] += x.T @ x

        total_samples = np.sum(n_samples)
        self.pi = n_samples / total_samples
        self.means = sum_x / n_samples[:, None]

        S_k = np.array(
            [
                (sum_x2[k] / n_samples[k]) - np.outer(self.means[k], self.means[k])
                for k in range(self.classes)
            ]
        )

        S_pooled = (
            np.sum([n_samples[k] * S_k[k] for k in range(self.classes)], axis=0)
            / total_samples
        )

        self.covariance_matrices = []
        for k in range(self.classes):
            S_reg = (1 - self.lmbda) * S_k[k] + self.lmbda * S_pooled

            if self.gamma > 0:
                average_eig = np.trace(S_reg) / self.in_features
                S_reg = (1 - self.gamma) * S_reg + self.gamma * average_eig * np.eye(
                    self.in_features
                )

            self.covariance_matrices.append(S_reg)

        self.inverse_covariance = np.array([inv(m) for m in self.covariance_matrices])
        print("training done")

        accuracy = self.validation(val_data)
        return accuracy


def run_RDA_training(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    val_matrix: np.ndarray,
    val_labels: np.ndarray,
    model_settings: dict,
):
    train_set = AnimalPictures(train_matrix, train_labels)
    val_set = AnimalPictures(val_matrix, val_labels)
    train_loader = DataLoader(train_set, batch_size=64)
    val_loader = DataLoader(val_set, batch_size=64)
    model = RegularizedDiscriminantAnalysis(**model_settings)

    score = model.train(train_loader, val_loader)

    return score


def main():
    train_matrix = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    train_matrix, val_matrix, train_labels, val_labels = train_test_split(
        train_matrix, train_labels, train_size=0.9, shuffle=True, stratify=train_labels
    )
    train_matrix = np.array(train_matrix)
    train_labels = np.array(train_labels)
    val_matrix = np.array(val_matrix)
    val_labels = np.array(val_labels)

    settings = {"in_features": 4096, "classes": 2, "lmbda": 0.1, "gamma": 0}

    score = run_RDA_training(
        train_matrix, train_labels, val_matrix, val_labels, settings
    )
    print(score)


if __name__ == "__main__":
    main()
