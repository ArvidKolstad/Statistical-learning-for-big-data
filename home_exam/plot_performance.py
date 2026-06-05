import matplotlib.pyplot as plt
from itertools import combinations
import numpy as np
from baycomp import two_on_single


def get_performance_idx(performance_score):
    if performance_score == "accuracy":
        index = 0
    elif performance_score == "f1-score":
        index = 1
    else:
        raise NotImplementedError
    return index


def get_rope_size(performance_score, runs):
    models = [
        f"KNN_{7680}.npy",
        f"LogReg_{7680}.npy",
        f"RDA_{7680}.npy",
    ]

    ropes = np.arange(0.0001, 0.2, 0.001)
    index = get_performance_idx(performance_score)
    comparisons = combinations(models, 2)

    for rope in ropes:
        saved_p_ropes = []
        comparisons = combinations(models, 2)
        for path1, path2 in comparisons:

            model_1 = np.load("models/" + path1)[:, index]
            model_2 = np.load("models/" + path2)[:, index]

            _, p_rope, _ = two_on_single(model_1, model_2, rope=rope, runs=runs)
            saved_p_ropes.append(p_rope)

        mean_p_ropes = np.mean(saved_p_ropes)
        if mean_p_ropes > 0.95:
            best_rope = rope
            print(f"Good rope was found: {best_rope}")
            return best_rope

    best_rope = 0.1
    print("Good rope wasn't found")
    return best_rope


def get_p_scores_compare(model_paths, performance_score, runs, rope=0.01):
    index = get_performance_idx(performance_score)
    models = []
    comparisons = combinations(model_paths, 2)

    for path1, path2 in comparisons:
        model_1 = np.load("models/" + path1)[:, index]
        model_2 = np.load("models/" + path2)[:, index]
        models.append(two_on_single(model_1, model_2, rope=rope, runs=runs))
    return models


def get_p_defect(normal_path, defect_paths, performance_score, runs, rope=0.01):
    index = get_performance_idx(performance_score)
    lefts, ropes, rights = [], [], []

    model_1 = np.load("models/" + normal_path)[:, index]
    for path2 in defect_paths:
        model_2 = np.load("models/" + path2)[:, index]
        p_left, p_rope, p_right = two_on_single(model_1, model_2, rope=rope, runs=runs)
        lefts.append(p_left)
        ropes.append(p_rope)
        rights.append(p_right)

    return lefts, ropes, rights


def load_data(paths):
    names = ["KNN", "LogReg", "RDA"]
    array_dict = {}
    for name, path in zip(names, paths):
        array_dict[name] = np.load("models/" + path)
    return array_dict


def plot_performance_2a():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    names = ["KNN", "LogReg", "RDA", "Hierarchy"]
    models = [
        f"2a/KNN_extreme.npy",
        f"2a/LogReg_extreme.npy",
        f"2a/RDA_extreme.npy",
        f"2a/HierarchyModel.npy",
    ]
    data = load_data(models)

    acc_data = [data[name][:, 0] for name in names]
    f1_data = [data[name][:, 1] for name in names]

    ax[0].boxplot(acc_data, tick_labels=names)
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy Distribution")
    ax[0].set_ylim(0.65, 1)
    ax[0].grid(axis="y", linestyle="--", alpha=0.7)

    ax[1].boxplot(f1_data, tick_labels=names)
    ax[1].set_ylabel("F1-Score")
    ax[1].set_title("F1-Score Distribution")
    ax[1].set_ylim(0.65, 1)
    ax[1].grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig("./figures/problem4/performance_boxplot.pdf")


def plot_performance(samples):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    acc_means = np.zeros((len(samples), 3))
    acc_std = np.zeros((len(samples), 3))

    f1_means = np.zeros((len(samples), 3))
    f1_std = np.zeros((len(samples), 3))

    names = ["KNN", "LogReg", "RDA"]

    for i, sample in enumerate(samples):
        models = [
            f"1a/KNN_{sample}.npy",
            f"1a/LogReg_{sample}.npy",
            f"1a/RDA_{sample}.npy",
        ]
        data = load_data(models)
        for j, name in enumerate(names):
            acc_means[i, j] = np.mean(data[name][:, 0])
            acc_std[i, j] = np.std(data[name][:, 0])

            f1_means[i, j] = np.mean(data[name][:, 1])
            f1_std[i, j] = np.std(data[name][:, 1])

    for idx in range(3):
        ax[0].errorbar(
            samples, acc_means[:, idx], yerr=acc_std[:, idx], label=names[idx]
        )
        ax[1].errorbar(samples, f1_means[:, idx], yerr=f1_std[:, idx], label=names[idx])

    ax[0].set_ylabel("Accuracy")
    ax[1].set_ylabel("F1-Score")

    ax[0].set_ylim(0.8, 1)
    ax[1].set_ylim(0.8, 1)

    ax[0].set_title("Accuracy")
    ax[1].set_title("F1-Score")

    ax[0].legend()
    ax[1].legend()

    fig.tight_layout()
    fig.savefig(f"./figures/problem1/performance.pdf")


def plot_bayes_data_destruction(sizes):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    ropes = [0.0056, 0.0107, 0.0062]
    names = ["KNN", "LogReg", "RDA"]
    f1_knn = []
    f1_log = []
    f1_rda = []
    f1_probs = [f1_knn, f1_log, f1_rda]

    for idx, name in enumerate(names):
        original_model = f"1a/{name}_7680.npy"
        disrupted_models = [f"1b/{name}_{size}.npy" for size in sizes]

        f1_left, f1_rope, f1_right = get_p_defect(
            original_model, disrupted_models, "f1-score", 10, rope=ropes[idx]
        )
        for i, size in enumerate(sizes):
            f1_stacks = {
                "P(Original > Disrupted)": f1_left[i],
                "P(Original = Disrupted)": f1_rope[i],
                "P(Original < Disrupted)": f1_right[i],
            }
            f1_probs[idx].append(f1_stacks)

    fig, ax = plt.subplots(1, 3, figsize=(23, 6))

    attributes = [
        "P(Original > Disrupted)",
        "P(Original = Disrupted)",
        "P(Original < Disrupted)",
    ]

    x = np.arange(len(sizes))
    width = 0.25

    for i in range(3):
        for attr_idx, attribute in enumerate(attributes):
            measurements = [f1_probs[i][j][attribute] for j in range(len(sizes))]

            offset = (attr_idx - 1) * width

            rects = ax[i].bar(
                x + offset,
                measurements,
                width,
                label=attribute,
                color=colors[attr_idx],
            )

            ax[i].bar_label(rects, padding=3, fmt="%.2f")

        ax[i].set_xticks(x)
        ax[i].set_xticklabels(sizes)

        ax[i].set_ylim(0, 1.15)
        ax[i].set_ylabel("Probability")
        ax[i].set_ylabel("Extra features")
        ax[i].set_title(names[i], fontsize=14)
        ax[i].legend()
        ax[i].grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig("./figures/problem2/bayes_comp.pdf")


def plot_bayes_analysis(samples):
    colors = ["#e74c3c", "#3498db", "#2ecc71"]

    f1_knn_log = []
    f1_knn_rda = []
    f1_knn_hier = []
    f1_log_rda = []
    f1_log_hier = []
    f1_rda_hier = []

    f1_scores = [
        f1_knn_log,
        f1_knn_rda,
        # f1_knn_hier,
        f1_log_rda,
        # f1_log_hier,
        # f1_rda_hier,
    ]

    for sample in samples:
        models = [
            f"1a/KNN_{sample}.npy",
            f"1a/LogReg_{sample}.npy",
            f"1a/RDA_{sample}.npy",
            # f"2a/HierarchyModel.npy",
        ]

        f1_models = get_p_scores_compare(models, "f1-score", 10, rope=0.040)
        for idx, model_comp in enumerate(f1_models):
            f1_left, f1_rope, f1_right = model_comp
            f1_stacks = {
                "P(Left > Right)": f1_left,
                "P(Left = Right)": f1_rope,
                "P(Left < Right)": f1_right,
            }
            f1_scores[idx].append(f1_stacks)

    names = [
        "KNN vs LogReg",
        "KNN vs RDA",
        # "KNN vs Hierarchy",
        "LogReg vs RDA",
        # "LogReg vs Hierarchy",
        # "RDA vs Hierarchy",
    ]
    fig, ax = plt.subplots(1, len(names), figsize=(11, 4))

    attributes = ["P(Left > Right)", "P(Left = Right)", "P(Left < Right)"]

    x = np.arange(len(samples))
    width = 0.25

    for i in range(len(names)):
        for attr_idx, attribute in enumerate(attributes):
            measurements = [f1_scores[i][j][attribute] for j in range(len(samples))]

            offset = (attr_idx - 1) * width

            rects = ax[i].bar(
                x + offset,
                measurements,
                width,
                label=attribute,
                color=colors[attr_idx],
            )

            ax[i].bar_label(rects, padding=3, fmt="%.2f")

        ax[i].set_xticks(x)
        ax[i].set_xticklabels(samples)

        ax[i].set_ylim(0, 1.15)
        ax[i].set_ylabel("Probability")
        ax[i].set_title(names[i], fontsize=14)
        ax[i].legend()
        ax[i].grid(axis="y", linestyle="--", alpha=0.7)

    fig.tight_layout()
    fig.savefig("./figures/problem1/bayes_comp.pdf")


def plot_class_accuracy(samples):
    names = ["KNN", "LogReg", "RDA"]
    samples_label = [f"{sample}" for sample in samples]
    classes = ["1", "2", "3", "4", "5", "6", "7"]

    statistics = np.zeros((4, 2, len(samples), 7))

    for j, sample in enumerate(samples):
        models = [
            f"1a/KNN_{sample}_fold_acc.npy",
            f"1a/LogReg_{sample}_fold_acc.npy",
            f"1a/RDA_{sample}_fold_acc.npy",
            # f"2a/HierarchyModel_fold_acc.npy",
        ]
        data_dict = load_data(models)
        for i, name in enumerate(names):
            statistics[i, 0, j, :] = np.mean(data_dict[name], axis=0)
            statistics[i, 1, j, :] = np.std(data_dict[name], axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    num_classes = len(classes)
    total_width = 0.8
    width = total_width / num_classes
    x = np.arange(len(samples))

    for i, name in enumerate(names):
        for c in range(num_classes):
            offset = (c - (num_classes - 1) / 2) * width

            ax[i].bar(
                x + offset,
                statistics[i, 0, :, c],
                yerr=statistics[i, 1, :, c],
                width=width,
                capsize=3,
                alpha=0.8,
                label=f"Class {classes[c]}",
            )

        ax[i].set_xlabel("Sample size")
        ax[i].set_ylabel("Accuracy")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(samples_label)
        ax[i].set_title(name)
        ax[i].grid(axis="y", linestyle="--", alpha=0.5)

    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=7)

    fig.tight_layout()
    fig.savefig("./figures/problem1/class_accuracy.pdf", bbox_inches="tight")


def main():
    # samples = ["extreme"]
    samples = [200, 500, 1000, 3000, 5000, 7680]
    sizes = [1, 3, 5, 15, 20, 40]

    plot_performance(samples)
    # plot_performance_2a()

    # get_rope_size("f1-score", 10)
    plot_bayes_analysis(samples)
    plot_class_accuracy(samples)
    # plot_bayes_data_destruction(sizes)


if __name__ == "__main__":

    main()
