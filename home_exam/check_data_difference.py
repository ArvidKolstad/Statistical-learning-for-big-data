import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score


class RandomForest:
    def __init__(self, **settings):
        self.model = RandomForestClassifier(**settings)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)


def main():

    classifier_settings = {
        "n_estimators": 100,
        "criterion": "gini",
        "max_depth": None,
        "min_samples_split": 30,
        "min_samples_leaf": 10,
        "min_weight_fraction_leaf": 0.0,
        "max_features": "sqrt",
        "min_impurity_decrease": 0.0,
        "bootstrap": True,
        "oob_score": True,
        "n_jobs": None,
        "random_state": None,
        "verbose": 0,
        "warm_start": False,
        "class_weight": "balanced",
        "ccp_alpha": 1.11e-4,
        "max_samples": 0.98,
    }
    inputs_class_zero = np.load("./data/7680_input.npy")
    inputs_class_one = np.load("./data/test_set.npy")

    label_zero = np.zeros(inputs_class_zero.shape[0])
    label_one = np.ones(inputs_class_one.shape[0])

    input_set = np.concatenate([inputs_class_zero, inputs_class_one], axis=0)
    label_set = np.concatenate([label_zero, label_one], axis=0)

    data_set = [input_set, label_set]

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    auc_score = []
    model = RandomForest()

    for _, (train_idx, val_idx) in enumerate(skf.split(*data_set)):
        train_batch = [values[train_idx] for values in data_set]
        val_batch = [values[val_idx] for values in data_set]
        model.fit(*train_batch)
        val_inputs, val_labels = val_batch
        preds = model.predict(val_inputs)
        auc_score.append(roc_auc_score(val_labels, preds))
    print(
        f"Mean AUC-score: {np.mean(auc_score):.3f} with standard deviation: {np.std(auc_score):.3f}"
    )


if __name__ == "__main__":
    main()
