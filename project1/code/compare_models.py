import numpy as np
import matplotlib.pyplot as plt

# For MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from multilayer_preceptron import ReducedDimDataset, MultilayerPerception
from dim_red import dimension_reduction
from kNNClassifier import tune_knn


# Comparisons
def main():

    # Load data
    train_data = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    test_data = np.load("./data/test_matrix.npy")
    test_labels = np.load("./data/test_labels.npy")

    # print("Train shape:", train_data.shape)
    # print("Test shape:", test_data.shape)
    # print("Label distribution:", np.unique(train_labels, return_counts=True))

    # Dimension reduction
    reduced_train, reduced_test = dimension_reduction(
        train_data,
        test_data=test_data,
        train_label=train_labels,
        n_dim_pca=50
    )

    # kNN
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(reduced_train, train_labels)

    # knn_pred = knn.predict(reduced_test)
    # knn_acc = accuracy_score(test_labels, knn_pred)

    knn = tune_knn(
        reduced_train,
        train_labels,
        k_values=range(1, 6),
        n_folds=2
    )

    knn_pred = knn.predict(reduced_test)
    knn_acc = accuracy_score(test_labels, knn_pred)

    # BUG (kNNClassifier.py):
    # All träningskod körs vid import → ger sid-effekter (plots)
    # Flytta allt till:
    # if __name__ == "__main__":

    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    rf.fit(reduced_train, train_labels)
    rf_pred = rf.predict(reduced_test)
    rf_acc = accuracy_score(test_labels, rf_pred)

    # NOTE: oob_score_ finns bara när bootstrap=True och oob_score=True

    # MLP
    # MLP
    mlp = MultilayerPerception(
        layer_dim=[reduced_train.shape[1], 128, 64, 10],
        act_func=["ReLU", "ReLU", "identity"],
        dropout_rate=0.2
    )

    mlp.fit(reduced_train, train_labels, epochs=5)

    mlp_pred = mlp.predict(reduced_test)
    mlp_acc = accuracy_score(test_labels, mlp_pred)
 
    # # Results
    # print("y_test shape:", test_labels.shape)
    # print("y_pred shape:", knn_pred.shape)
    # print("unique labels:", np.unique(test_labels))
    # print("unique preds:", np.unique(knn_pred))

    results = {
        "kNN": knn_acc,
        "Random Forest": rf_acc,
        "MLP": mlp_acc,
    }

    for name, acc in results.items():
        print(f"{name}: {acc:.4f}")

    # Plotting
    plt.bar(results.keys(), results.values())
    plt.ylabel("Accuracy")
    plt.title("Model comparison on MNIST")
    plt.show()


if __name__ == "__main__":
    main()