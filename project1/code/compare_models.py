import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from multilayer_preceptron import (
    load_mlp_model,
)  # ReducedDimDataset, MultilayerPerception
from dim_red import dimension_reduction
from knn_classifier import tune_knn_and_dim_red
from random_forest import MyRandomForest  # train_rfc
from config_rf import classifier_settings
from logistic_classifier import tune_dim_red


# Results Function
def find_results(name, predictions, test_labels, save_path=None):
    accuracy = accuracy_score(test_labels, predictions)
    errors = (predictions != test_labels).sum()

    cm = confusion_matrix(test_labels, predictions)

    fig, ax = plt.subplots(figsize=(7, 6))

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)

    ax.set_title(f"{name}Confusion Matrix")
    plt.tight_layout()
    plt.show()

    # Save Figure
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.close(fig)

    return accuracy, errors


# Comparisons
def main():

    # Load data
    train_data = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    test_data = np.load("./data/test_matrix.npy")
    test_labels = np.load("./data/test_labels.npy")

    # kNN
    knn, best_dim = tune_knn_and_dim_red(
        train_data,
        train_labels,
        k_values=range(1, 6),
        n_folds=2,
        n_dims=range(10, 101, 10),
    )

    _, reduced_test = dimension_reduction(
        train_data, test_data=test_data, n_dim_pca=best_dim
    )

    knn_pred = knn.predict(reduced_test)

    # save_path_knn = "../figures/cm_knn.png"
    # knn_acc, knn_err = find_results(
    #     "kNN ",
    #     knn_pred,
    #     test_labels,
    #     save_path_knn)

    knn_acc, knn_err = find_results("kNN ", knn_pred, test_labels)

    # Random Forest
    rf = MyRandomForest().load("./saved_models/random_forest")
    best_dim_rf = int(np.load("./saved_models/random_forest_dim.npy"))

    _, reduced_test_rf = dimension_reduction(
        train_data, test_data=test_data, n_dim_pca=best_dim_rf
    )

    rf_pred = rf.predict(reduced_test_rf)

    # save_path_rf = "../figures/cm_rf.png"
    # rf_acc, rf_err = find_results(
    #     "Random Forest ",
    #     rf_pred,
    #     test_labels,
    #     save_path_rf)

    rf_acc, rf_err = find_results("Random Forest ", rf_pred, test_labels)

    # # MLP
    mlp = load_mlp_model("./saved_models/mlp_settings", "./saved_models/mlp")

    _, reduced_test_mlp = dimension_reduction(
        train_data, test_data=test_data, train_label=train_labels, n_dim_pca=44
    )

    mlp.to(mlp.device)
    mlp_pred = mlp.predict(reduced_test_mlp)

    # save_path_mlp = "../figures/cm_mlp.png"
    # mlp_acc, mlp_err = find_results(
    #     "MLP ",
    #     mlp_pred,
    #     test_labels,
    #     save_path_mlp)

    mlp_acc, mlp_err = find_results("MLP ", mlp_pred, test_labels)

    # Logistic Regression
    lr, best_dim_lr, scaler = tune_dim_red(
        train_data, train_labels, n_dims=range(10, 151, 10), n_folds=2
    )

    _, reduced_test_lr = dimension_reduction(
        train_data, test_data=test_data, n_dim_pca=best_dim_lr
    )

    scaled_reduced_test_lr = scaler.transform(reduced_test_lr)

    lr_pred = lr.predict(scaled_reduced_test_lr)

    # save_path_lr = "../figures/cm_lr.png"
    # lr_acc, lr_err = find_results(
    #     "Logistic Regression ",
    #     lr_pred,
    #     test_labels,
    #     save_path_lr)

    lr_acc, lr_err = find_results("Logistic Regression ", lr_pred, test_labels)

    # Results
    results_acc = {
        "kNN": knn_acc,
        "Random Forest": rf_acc,
        "MLP": mlp_acc,
        "LogReg": lr_acc,
    }

    results_err = {
        "kNN": knn_err,
        "Random Forest": rf_err,
        "MLP": mlp_err,
        "LogReg": lr_err,
    }

    print("\nAccuracy")
    for name, acc in results_acc.items():
        print(f"{name:<15}: {acc:.4f}")

    print(f"\nNumber of errors out of {len(test_labels)}")
    for name, err in results_err.items():
        print(f"{name:<15}: {err}")


if __name__ == "__main__":
    main()
