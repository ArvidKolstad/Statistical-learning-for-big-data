import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from multilayer_preceptron import ReducedDimDataset, MultilayerPerception
from dim_red import dimension_reduction
from knn_classifier import tune_knn_and_dim_red
from random_forest import train_rfc
from config_rf import classifier_settings
from logistic_classifier import tune_dim_red



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
    knn, best_dim = tune_knn_and_dim_red(
        train_data,
        train_labels,
        k_values=range(1, 6),
        n_folds=2,
        n_dims=range(10, 101, 10)
    )

    reduced_train, reduced_test = dimension_reduction(
        train_data,
        test_data=test_data,
        n_dim_pca=best_dim
    )

    knn_pred = knn.predict(reduced_test)
    knn_acc, knn_err = find_results("kNN ", knn_pred, test_labels)


    # Random Forest
    rf = train_rfc(
        reduced_train,
        train_labels,
        classifier_settings
    )

    # rf.fit(reduced_train, train_labels)
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


    # Logistic Regression
    lr, best_dim_lr, scaler = tune_dim_red(
        train_data,
        train_labels,
        n_dims=range(10, 151, 10),
        n_folds=2
    )

    reduced_train_lr, reduced_test_lr = dimension_reduction(
        train_data,
        test_data=test_data,
        n_dim_pca=best_dim_lr
    )

    reduced_test_lr = scaler.transform(reduced_test_lr)

    lr_pred = lr.predict(reduced_test_lr)
    lr_acc, lr_err = find_results("Logistic Regression ", lr_pred, test_labels)


    # Results
    results_acc = {
        "kNN": knn_acc,
        "Random Forest": rf_acc,
        "MLP": mlp_acc,
        "LogReg": lr_acc
    }

    results_err = {
        "kNN": knn_err,
        "Random Forest": rf_err,
        "MLP": mlp_err,
        "LogReg": lr_err
    }

    print("\nAccuracy")
    for name, acc in results_acc.items():
        print(f"{name:<15}: {acc:.4f}")

    print(f"\nNumber of errors out of {len(test_labels)}")
    for name, err in results_err.items():
        print(f"{name:<15}: {err}")



if __name__ == "__main__":
    main()