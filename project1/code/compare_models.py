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


# Model training for MLP
def evaluate_mlp(X_train, y_train, X_test, y_test):
    dataset = ReducedDimDataset(X_train, y_train)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = MultilayerPerception(
        layer_dim=[X_train.shape[1], 128, 64, 10],
        act_func=["ReLU", "ReLU", "identity"],
        dropout_rate=0.2
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  
        for x, y in train_loader:
            optimizer.zero_grad()
            out = model(x.float())
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(X_test).float()
        outputs = model(X_test_t)
        preds = torch.argmax(outputs, dim=1).numpy()

    return accuracy_score(y_test, preds)


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
    mlp_acc = evaluate_mlp(
        reduced_train, train_labels,
        reduced_test, test_labels
    )

    # BUG (multilayer_preceptron.py):
    # labels måste vara torch.long för CrossEntropyLoss
    # outputs.squeeze(1) i validate_model() är felaktigt för classification logits

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