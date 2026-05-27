import matplotlib.pyplot as plt
from performance_score import get_p_scores
import numpy as np


def load_data(paths):
    names = ["KNN", "LogReg", "XGB", "RDA"]
    array_dict = {}
    for name, path in zip(names, paths):
        array_dict[name] = np.load("models/" + path)
    return array_dict


def plot_performance(paths, samples):

    data = load_data(paths)
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    names = ["KNN", "LogReg", "XGB", "RDA"]
    accuracy = np.vstack(
        [data["KNN"][:, 0], data["LogReg"][:, 0], data["XGB"][:, 0], data["RDA"][:, 0]]
    )
    f1 = np.vstack(
        [data["KNN"][:, 1], data["LogReg"][:, 1], data["XGB"][:, 1], data["RDA"][:, 1]]
    )

    aoc = np.vstack(
        [data["KNN"][:, 2], data["LogReg"][:, 2], data["XGB"][:, 2], data["RDA"][:, 2]]
    )
    ax[0].boxplot(accuracy.T, tick_labels=names)
    ax[1].boxplot(f1.T, tick_labels=names)
    ax[2].boxplot(aoc.T, tick_labels=names)

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("F1-Score")
    ax[2].set_ylabel("AOC")

    ax[0].set_ylim(0.4, 1)
    ax[1].set_ylim(0.4, 1)
    ax[2].set_ylim(0.4, 1)

    ax[0].set_title("Accuracy")
    ax[1].set_title("F1-Score")
    ax[2].set_title("AOC-curve")

    fig.tight_layout()
    fig.savefig(f"figures_part1/preformace_{samples}")


def plot_bayes_analysis(model_paths, samples):

    acc_left, acc_rope, acc_right = get_p_scores(
        model_paths, "accuracy", 10, rope=0.052
    )
    f1_left, f1_rope, f1_right = get_p_scores(model_paths, "f1-score", 10, rope=0.0821)
    aoc_left, aoc_rope, aoc_right = get_p_scores(model_paths, "AOC", 10, rope=0.052)

    acc_stacks = {
        "P(Left model > Right model)": acc_left,
        "P(Left model = Right model)": acc_rope,
        "P(Right model < Right model)": acc_right,
    }
    f1_stacks = {
        "P(Left model > Right model)": f1_left,
        "P(Left model = Right model)": f1_rope,
        "P(Right model < Right model)": f1_right,
    }
    aoc_stacks = {
        "P(Left model > Right model)": aoc_left,
        "P(Left model = Right model)": aoc_rope,
        "P(Right model < Right model)": aoc_right,
    }

    fig, ax = plt.subplots(1, 3, figsize=(23, 5))

    names = [
        "KNN vs LogReg",
        "KNN vs XGB",
        "KNN vs RDA",
        "LogReg vs XGB",
        "LogReg vs RDA",
        "XGB vs RDA",
    ]
    x = np.arange(len(names))
    width = 0.2
    multiplier = 0
    for attribute, measurement in acc_stacks.items():
        offset = width * multiplier
        rects = ax[0].bar(x + offset, measurement, width, label=attribute)
        # ax[0].bar_label(rects, padding=1)
        multiplier += 1

    multiplier = 0
    for attribute, measurement in f1_stacks.items():
        offset = width * multiplier

        rects = ax[1].bar(x + offset, measurement, width, label=attribute)
        # ax[1].bar_label(rects, padding=1)
        multiplier += 1

    multiplier = 0
    for attribute, measurement in aoc_stacks.items():
        offset = width * multiplier

        rects = ax[2].bar(x + offset, measurement, width, label=attribute)
        # ax[2].bar_label(rects, padding=1)
        multiplier += 1

    ax[0].set_xticks(x + width, names)
    ax[1].set_xticks(x + width, names)
    ax[2].set_xticks(x + width, names)

    ax[0].set_ylim(0, 1.3)
    ax[1].set_ylim(0, 1.3)
    ax[2].set_ylim(0, 1.3)

    ax[0].set_ylabel("Probability")
    ax[1].set_ylabel("Probability")
    ax[2].set_ylabel("Probability")

    ax[0].set_title("Accuracy")
    ax[1].set_title("F1-Score")
    ax[2].set_title("AOC")

    ax[0].legend()
    ax[1].legend()
    ax[2].legend()

    fig.tight_layout()
    fig.savefig(f"./figures_part2/barchart_{samples}.png")


def main():
    samples = 500
    model_paths = [
        f"KNN_{samples}.npy",
        f"LogReg_{samples}.npy",
        f"XGB_{samples}.npy",
        f"RDA_{samples}.npy",
    ]
    plot_performance(model_paths, samples)
    # plot_bayes_analysis(model_paths, samples)


if __name__ == "__main__":

    main()
