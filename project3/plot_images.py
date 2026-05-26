import matplotlib.pyplot as plt
from performance_score import get_p_scores
import numpy as np


def load_data(paths):
    names = ["logreg", "KNN", "XGB", "RDA"]
    array_dict = {}
    for name, path in zip(names, paths):
        array_dict[name] = np.load(path)
    return array_dict


def plot_performance(data, samples):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    names = ["logreg", "KNN", "XGB", "RDA"]
    accuracy = np.vstack(
        [data["logreg"][:, 0], data["KNN"][:, 0], data["XGB"][:, 0], data["RDA"][:, 0]]
    )
    f1 = np.vstack(
        [data["logreg"][:, 1], data["KNN"][:, 1], data["XGB"][:, 1], data["RDA"][:, 1]]
    )

    aoc = np.vstack(
        [data["logreg"][:, 2], data["KNN"][:, 2], data["XGB"][:, 2], data["RDA"][:, 2]]
    )
    ax[0].boxplot(accuracy, labels=names)
    ax[1].boxplot(f1, labels=names)
    ax[2].boxplot(aoc, labels=names)

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("F1-Score")
    ax[2].set_ylabel("AOC")

    ax[0].set_titel("Accuracy")
    ax[1].set_titel("F1-Score")
    ax[2].set_titel("AOC-curve")

    fig.tight_layout()
    fig.savefig(f"figures_part1/preformace_{samples}")


def plot_p_matrix(model_paths, samples):

    accuracy = get_p_scores(model_paths, "accuracy", 10)
    f1 = get_p_scores(model_paths, "f1-score", 10)
    aoc = get_p_scores(model_paths, "AOC", 10)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    names = ["logreg", "KNN", "XGB", "RDA"]

    ax[0].matshow(accuracy)
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names)

    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names)

    ax[1].matshow(f1)
    ax[1].set_xticks(range(len(names)))
    ax[1].set_xticklabels(names)

    ax[1].set_yticks(range(len(names)))
    ax[1].set_yticklabels(names)

    ax[2].mathshow(aoc)
    ax[2].set_xticks(range(len(names)))
    ax[2].set_xticklabels(names)

    ax[2].set_yticks(range(len(names)))
    ax[2].set_yticklabels(names)

    ax[0].set_titel("Accuracy")
    ax[1].set_titel("F1-Score")
    ax[2].set_titel("AOC-curve")

    fig.suptitle("P(model(i) > model(j))")
    fig.tight_layout()
    fig.savefig(f"figures_part1/p_scores_{samples}.pdf")


def main():
    print("Hello")
    samples = 1000
    model_paths = [
        f"LogReg_{samples}.npy",
        f"KNN_{samples}.npy",
        f"XGB_{samples}.npy",
        f"RDA_{samples}.npy",
    ]
    data = load_data(model_paths)


if __name__ == "__main__":

    main()
