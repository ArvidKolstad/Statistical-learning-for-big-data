import numpy as np
import matplotlib.pyplot as plt

# For MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier

from multilayer_preceptron import ReducedDimDataset, MultilayerPerception
from dim_red import dimension_reduction
from kNNClassifier import tune_knn
from random_forest import train_rfc
from config_rf import classifier_settings


# Results Function
def find_results(name, predictions, test_labels):
    accuracy = accuracy_score(test_labels, predictions)
    errors = (predictions != test_labels).sum()

    cm = confusion_matrix(test_labels, predictions)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap="Blues")
    plt.title(f"{name}Confusion Matrix")
    plt.show()

    return accuracy, errors


# Comparisons
def main():

    # Load data
    train_data = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    test_data = np.load("./data/test_matrix.npy")
    test_labels = np.load("./data/test_labels.npy")

    # Dimension reduction
    reduced_train, reduced_test = dimension_reduction(
        train_data,
        test_data=test_data,
        train_label=train_labels,
        n_dim_pca=50
    )

    # kNN
    knn = tune_knn(
        reduced_train,
        train_labels,
        k_values=range(1, 6),
        n_folds=2
    )

    knn_pred = knn.predict(reduced_test)
    knn_acc, knn_err = find_results("kNN ", knn_pred, test_labels)


    # Random Forest
    rf = train_rfc(
        reduced_train,
        train_labels,
        classifier_settings
    )

    rf.fit(reduced_train, train_labels)
    rf_pred = rf.predict(reduced_test)
    rf_acc, rf_err = find_results("Random Forest ", rf_pred, test_labels)


    # MLP
    mlp = MultilayerPerception(
        layer_dim=[reduced_train.shape[1], 128, 64, 10],
        act_func=["ReLU", "ReLU", "identity"],
        dropout_rate=0.2
    )

    mlp.fit(reduced_train, train_labels, epochs=5)
    mlp_pred = mlp.predict(reduced_test)
    mlp_acc, mlp_err = find_results("MLP ", mlp_pred, test_labels)


    # Results
    results_acc = {
        "kNN": knn_acc,
        "Random Forest": rf_acc,
        "MLP": mlp_acc,
    }

    results_err = {
        "kNN": knn_err,
        "Random Forest": rf_err,
        "MLP": mlp_err
    }

    print("\nAccuracy")
    for name, acc in results_acc.items():
        print(f"{name:<15}: {acc:.4f}")

    print(f"\nNumber of errors out of {len(test_labels)}")
    for name, err in results_err.items():
        print(f"{name:<15}: {err}")



if __name__ == "__main__":
    main()