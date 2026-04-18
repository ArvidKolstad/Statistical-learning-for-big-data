import numpy as np
import matplotlib.pyplot as plt

# For MLP
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from multilayer_preceptron import ReducedDimDataset, MultilayerPerception
from dim_red import dimension_reduction
from kNNClassifier import tune_knn


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
    knn_acc = accuracy_score(test_labels, knn_pred)


    # Random Forest
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        random_state=42
    )

    rf.fit(reduced_train, train_labels)
    rf_pred = rf.predict(reduced_test)

    # rf_acc = accuracy_score(test_labels, rf_pred)

    # # Random Forest Confusion Matrix
    # cm_rf = confusion_matrix(test_labels, rf_pred)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    # disp.plot(cmap="Blues")
    # plt.title("Random Forest Confusion Matrix")
    # plt.show()

    # # Antal fel
    # rf_errors = (rf_pred != test_labels).sum()

    rf_acc, rf_err = find_results("Random Forest ", rf_pred, test_labels)
    print(rf_acc)
    print(rf_err)

    # NOTE: oob_score_ finns bara när bootstrap=True och oob_score=True

    # MLP
    # MLP
    mlp = MultilayerPerception(
        layer_dim=[reduced_train.shape[1], 128, 64, 10],
        act_func=["ReLU", "ReLU", "identity"],
        dropout_rate=0.2
    )

    mlp.fit(reduced_train, train_labels, epochs=5)

    # Accuary 
    mlp_pred = mlp.predict(reduced_test)
    # mlp_acc = accuracy_score(test_labels, mlp_pred)

    # # MLP Confusion matrix
    # cm_mlp = confusion_matrix(test_labels, mlp_pred)

    # disp = ConfusionMatrixDisplay(confusion_matrix=cm_mlp)
    # disp.plot(cmap="Blues")
    # plt.title("MLP Confusion Matrix")
    # plt.show()
 
    # # Antal fel
    # mlp_errors = (mlp_pred != test_labels).sum()

    mlp_acc, mlp_err = find_results("MLP ", mlp_pred, test_labels)
    print(mlp_acc)
    print(mlp_err)
   
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