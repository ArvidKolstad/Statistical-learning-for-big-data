from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
from numpy.linalg import inv, slogdet
from train_utils import AnimalPictures, kCV


class RegularizedDiscriminantAnalysis:
    def __init__(
        self,
        in_features: int,
        classes: int,
        lmbda: float,
        gamma: float,
        class_names: list,
        load_params=None,
    ) -> None:
        self.in_features = in_features
        self.classes = classes

        self.lmbda = lmbda
        self.gamma = gamma

        self.mean_vector = np.zeros((classes, in_features))
        self.covariance_matrices = np.zeros((classes, in_features, in_features))
        self.inverse_covariances = np.zeros((classes, in_features, in_features))
        self.pi = np.zeros(classes)
        self.class_names = class_names

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

    def validation(self, val_loader: DataLoader, save_confusion_matrix=None) -> float:
        X_all = np.concatenate([X.numpy() for X, _ in val_loader], axis=0)
        labels_all = np.concatenate([y.numpy() for _, y in val_loader], axis=0)

        pred = self.decision_rule(X_all)
        correct_classification = np.sum(labels_all == pred)
        total_classification = labels_all.shape[0]
        if save_confusion_matrix:
            conf_mat = confusion_matrix(labels_all, pred, normalize="all")
            disp = ConfusionMatrixDisplay(conf_mat, display_labels=self.class_names)
            disp.plot()
            disp.figure_.savefig(save_confusion_matrix)

        accuracy = correct_classification / total_classification
        return accuracy

    def train(
        self, train_data: DataLoader, val_data: DataLoader, save_confusion_matrix=None
    ):
        n_samples = np.zeros(self.classes)
        sum_x = np.zeros((self.classes, self.in_features))
        sum_x2 = np.zeros((self.classes, self.in_features, self.in_features))

        for X_batch, labels in train_data:
            print(np.sum(n_samples))
            X_batch, labels = X_batch.numpy(), labels.numpy()
            for k in range(self.classes):
                mask = labels == k
                if not mask.any():
                    continue
                x = X_batch[mask]
                n_samples[k] += len(x)
                sum_x[k] += x.sum(axis=0)
                sum_x2[k] += x.T @ x

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

        self.inverse_covariances = inv(S_reg)

        return self.validation(val_data, save_confusion_matrix=save_confusion_matrix)


def run_RDA_training(
    model: RegularizedDiscriminantAnalysis,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    val_matrix: np.ndarray,
    val_labels: np.ndarray,
):
    train_set = AnimalPictures(train_matrix, train_labels)
    val_set = AnimalPictures(val_matrix, val_labels)
    train_loader = DataLoader(train_set, batch_size=64)
    val_loader = DataLoader(val_set, batch_size=64)

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
    val_loader = DataLoader(AnimalPictures(val_matrix, val_labels), batch_size=64)

    settings = {
        "in_features": 4096,
        "classes": 2,
        "lmbda": 0.1,
        "gamma": 0,
        "class_names": ["Cats", "Dogs"],
    }
    train_settings = {"train_data": None, "val_data": None}
    model = RegularizedDiscriminantAnalysis(**settings)

    # score = run_RDA_training(model, train_matrix, train_labels, val_matrix, val_labels)
    # model.save("./saved_models/RDA_first_try")
    # print(f"Score: {score}")
    model.load("./saved_models/RDA_first_try.npz")
    model.validation(
        val_loader, save_confusion_matrix="../figures/RDA/confusion_mat.png"
    )

    kCV(
        2,
        RegularizedDiscriminantAnalysis,
        train_matrix,
        train_labels,
        settings,
        train_settings,
        data_loaders=True,
    )


if __name__ == "__main__":
    main()
