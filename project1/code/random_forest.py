import numpy as np
from dim_red import dimension_reduction
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

# Wrapper
class MyRandomForest:
    def __init__(self, **settings):
        self.model = RandomForestClassifier(**settings)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def save(self, path):
        with open(path, "wb") as f:
            pkl.dump(self.model, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.model = pkl.load(f)
        return self


def train_rfc(
    inputs: np.ndarray,
    labels: np.ndarray,
    settings: dict,
    save_model=None,
):

    # classifier = RandomForestClassifier(**settings)
    # classifier.fit(inputs, labels)

    # classification_score = classifier.oob_score_

    rf = MyRandomForest(**settings)
    rf.fit(inputs, labels)

    if save_model:
        # params = classifier.get_params()
        # with open(save_model, "wb") as handle:
        #     pkl.dump(params, handle)
        # with open(save_model + "settings", "wb") as handle:
        #     pkl.dump(settings, handle)
        rf.save(save_model)

    return rf #classification_score


def find_good_ccp_alpha(input_matrix, labels, settings) -> float:
    print("Now finding good alpha")
    tree = DecisionTreeClassifier()
    path = tree.cost_complexity_pruning_path(input_matrix, labels)
    alphas = path.ccp_alphas
    scores = []
    settings["oob_score"] = False
    for idx, alpha in enumerate(alphas):
        print(f"Done {idx+1}/{len(alphas)}")
        settings["ccp_alpha"] = alpha
        rf = RandomForestClassifier(**settings)
        score = cross_val_score(rf, input_matrix, labels).mean()
        scores.append(score)
    best_alpha = alphas[np.argmax(scores)]
    settings["oob_score"] = True

    return best_alpha


def find_good_max_sample(
    input_matrix: np.ndarray,
    labels: np.ndarray,
    sample_range: list[float],
    settings: dict,
):
    original_max_samples = settings["max_samples"]
    sample_frac = np.arange(sample_range[0], sample_range[1], 0.01)
    sample_len = len(sample_frac)
    scores = []
    for idx, frac in enumerate(sample_frac):
        print(f"Done {idx+1}/{sample_len}")
        settings["max_samples"] = frac
        rf = RandomForestClassifier(**settings)
        score = cross_val_score(rf, input_matrix, labels).mean()
        scores.append(score)
    best_sample_frac = sample_frac[np.argmax(scores)]

    print(f"Best sample frac found: {best_sample_frac}")

    settings["max_samples"] = original_max_samples

    return best_sample_frac


def plot_accuracy_rate(
    inputs: np.ndarray,
    labels: np.ndarray,
    max_number_of_trees: int,
    settings: dict,
    save_plot: str,
):
    plot_scores = []
    original_n_estimator = settings["n_estimators"]
    for n_trees in range(1, max_number_of_trees + 1):
        settings["n_estimators"] = n_trees
        plot_scores.append(train_rfc(inputs, labels, settings))

        if n_trees % 10 == 0:
            print(f"Training done: {n_trees}/{max_number_of_trees}")

    fig, ax = plt.subplots(dpi=300)
    n_trees = np.arange(1, max_number_of_trees + 1, 1)
    ax.scatter(n_trees, plot_scores)
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy ploted over number of trees for RF-classifier")
    ax.grid()
    fig.tight_layout()
    fig.savefig(save_plot)
    settings["n_estimators"] = original_n_estimator


def main():
    dimensions = 3
    max_n_trees = 200
    save_plot = "../figures/RF_accuracy_over_many_trees.png"
    max_samples_range = [0.6, 1.0]

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
        "ccp_alpha": 1.09e-4,
        "max_samples": 0.86,
    }

    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    training_matrix, _ = dimension_reduction(
        training_matrix, train_label=training_labels)

    """
    best_ccp_alpha = find_good_ccp_alpha(
        training_matrix, training_labels, classifier_settings
    )
    print(f"Best alpha: {best_ccp_alpha}")
    classifier_settings["ccp_alpha"] = best_ccp_alpha

    # print(training_matrix.shape)
    find_good_max_sample(
        training_matrix, training_labels, max_samples_range, classifier_settings
    )
    plot_accuracy_rate(
        training_matrix,
        training_labels,
        max_n_trees,
        classifier_settings,
        save_plot,
    )
    """
    score = train_rfc(
        training_matrix,
        training_labels,
        classifier_settings,
        save_model="./saved_models/random_forest",
    )
    # print(score)
    print("RF trained")


if __name__ == "__main__":
    main()
