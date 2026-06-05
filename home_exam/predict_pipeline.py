import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from train_pipeline import BaseModelAdapter


def predict_testset(
    model_adapter: BaseModelAdapter, train_data: list[np.ndarray], test_data: np.ndarray
):
    model_adapter.clean_model()

    model_adapter.train_params(train_data)
    pred_classes = model_adapter.get_class(test_data)
    pred_prob = model_adapter.get_probability(test_data)
    _, train_labels = train_data
    unique_labels = np.unique(train_labels)
    class_prob_stats = np.zeros((len(unique_labels), 2))
    mean_prob = np.mean(np.max(pred_prob, axis=1))

    for cls in unique_labels:
        mask = cls == pred_classes
        class_preds = pred_prob[mask, :]
        preded_class = np.argmax(class_preds, axis=1)
        for p in preded_class:
            if not cls == p:
                raise (ValueError("class_preds and predicted class is not the same"))
        class_prob_stats[cls, 0] = np.mean(np.max(class_preds, axis=1))
        class_prob_stats[cls, 1] = np.std(np.max(class_preds, axis=1))
    np.save(model_adapter.output_dir + "stats", class_prob_stats)
    np.save(model_adapter.output_dir + "mean", mean_prob)
    np.save(model_adapter.output_dir + "preded_class", pred_classes)


def get_confidence_accuracy(model_confidence, model_preds, model_labels):
    bin_ends = np.arange(0.20, 1.05, 0.05)
    n_bins = len(bin_ends) - 1
    bins = np.zeros(n_bins)
    freq = np.zeros(n_bins)
    total_samples = len(model_confidence)

    ece = 0.0
    for idx in range(n_bins):
        start_bin = bin_ends[idx]
        end_bin = bin_ends[idx + 1]
        if idx == n_bins - 1:
            mask = (model_confidence >= start_bin) & (model_confidence <= end_bin)
        else:
            mask = (model_confidence >= start_bin) & (model_confidence < end_bin)

        freq[idx] = np.sum(mask)

        if freq[idx] > 0:
            bins[idx] = np.sum(model_labels[mask] == model_preds[mask]) / freq[idx]

            bin_mean_conf = np.mean(model_confidence[mask])

            bin_error = np.abs(bins[idx] - bin_mean_conf)

            bin_weight = freq[idx] / total_samples
            ece += bin_weight * bin_error
        else:
            bins[idx] = 0.0

    return bins, freq, ece


def get_model_probability_performance(model_adapter, train_data, splits):
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=42)
    model_confidence = []
    model_preds = []
    model_labels = []

    for _, (train_idx, val_idx) in enumerate(skf.split(*train_data)):
        train_batch = [values[train_idx] for values in train_data]
        val_batch = [values[val_idx] for values in train_data]
        model_adapter.clean_model()
        model_adapter.train_params(train_batch)

        preds, labels = model_adapter.validate(val_batch)
        model_preds.append(preds)
        model_labels.append(labels)
        val_input, _ = val_batch
        model_confidence.append(
            np.max(model_adapter.get_probability(val_input), axis=1)
        )

    model_confidence = np.concatenate(model_confidence, axis=0)
    model_preds = np.concatenate(model_preds, axis=0)
    model_labels = np.concatenate(model_labels, axis=0)

    bins, freq, ece = get_confidence_accuracy(
        model_confidence, model_preds, model_labels
    )
    np.save(model_adapter.output_dir + "bins", bins)
    np.save(model_adapter.output_dir + "freq", freq)
    np.save(model_adapter.output_dir + "ece", ece)


def plot_confidence_acc():
    model_name = ["KNN", "RDA"]

    bin_ends = np.arange(0.20, 1.05, 0.05)

    samples = [200, 500, 1000, 3000, 5000, 7680]
    for sample in samples:
        fig, ax1 = plt.subplots(1, 2, figsize=(10, 5))

        for idx, model in enumerate(model_name):

            bins = np.load(f"./models/1d/{model}_{sample}bins.npy")
            freq = np.load(f"./models/1d/{model}_{sample}freq.npy")

            x = np.arange(len(bins))
            width = 0.30
            gap = 0.05

            ax2 = ax1[idx].twinx()
            line_plot = ax1[idx].plot(
                np.arange(len(bin_ends)),
                bin_ends,
                label="Perfect Predictions",
                alpha=0.3,
                linestyle="--",
                color="gray",
            )[0]

            bars1 = ax1[idx].bar(
                x - width / 2 - gap / 2,
                bins,
                width=width,
                color="steelblue",
                alpha=0.8,
                label="Actual accuracy",
            )
            bars2 = ax2.bar(
                x + width / 2 + gap / 2,
                freq,
                width,
                color="tomato",
                alpha=0.8,
                label="Frequency",
            )
            ax1[idx].bar_label(
                bars1, fmt="%.2f", padding=3, fontsize=7, color="steelblue"
            )
            ax2.bar_label(bars2, fmt="%.0f", padding=3, fontsize=7, color="tomato")

            ax1[idx].set_xlabel("Model Confidence")
            ax1[idx].set_ylabel("Actual accuracy", color="steelblue")
            ax2.set_ylabel("Frequency", color="tomato")
            ax1[idx].tick_params(axis="y", labelcolor="steelblue")
            ax2.tick_params(axis="y", labelcolor="tomato")

            ax1[idx].set_xticks(x)

            labels = [
                f"{bin_ends[b_idx]:.2f} - {bin_ends[b_idx+1]:.2f}"
                for b_idx in range(len(bins))
            ]

            h1, l1 = ax1[idx].get_legend_handles_labels()
            h2, l2 = ax2.get_legend_handles_labels()
            ax1[idx].legend(h1 + h2, l1 + l2, loc="upper left")
            ax1[idx].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        ax1[0].set_title("KNN")
        ax1[1].set_title("RDA")

        fig.suptitle("Model Confidence Accuracy")
        fig.tight_layout()
        fig.savefig(f"./figures/problem3/model_conf_acc_{sample}.pdf")


def plot_confidence_over_sample_diff():
    model_name = ["KNN", "RDA"]
    samples = [200, 500, 1000, 3000, 5000, 7680]
    ece = np.zeros((len(samples), 2))

    fig, ax = plt.subplots()

    for model_idx, model in enumerate(model_name):
        for sample_idx, sample in enumerate(samples):
            ece[sample_idx, model_idx] = np.load(f"./models/1d/{model}_{sample}ece.npy")

        ax.plot(samples, ece[:, model_idx], label=model)

    ax.set_xlabel("Sample size")
    ax.set_ylabel("Expected Calibration Error")
    ax.legend()
    ax.set_title("Expected Calibration Error over Sample Size")
    fig.tight_layout()
    fig.savefig("./figures/problem3/conf_diff_sample_size.pdf")


def plot_confidence_for_test():
    model_name = ["KNN", "RDA"]
    samples = [200, 500, 1000, 3000, 5000, 7680]
    samples_label = [f"{sample}" for sample in samples]

    classes = ["1", "2", "3", "4", "5", "6", "7"]
    statistics = np.zeros((2, 2, len(samples), 7))
    mean_predict = np.zeros((2, len(samples)))

    for model_idx, model in enumerate(model_name):
        for sample_idx, sample in enumerate(samples):
            path = f"./models/1d/{model}_{sample}stats.npy"
            statistics[model_idx, :, sample_idx, :] = np.load(path).T
            mean_predict[model_idx, sample_idx] = np.load(
                f"./models/1d/{model}_{sample}mean.npy"
            )

    fig, ax = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    num_classes = len(classes)
    total_width = 0.8
    width = total_width / num_classes
    x = np.arange(len(samples))

    for i, name in enumerate(model_name):
        total_accuracy = mean_predict[i]

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

        ax[i].plot(
            x,
            total_accuracy,
            color="black",
            linestyle="--",
            linewidth=2.5,
            marker="o",
            markersize=6,
            label="Mean Confidence",
            zorder=3,
        )

        ax[i].set_xlabel("Sample size")
        ax[i].set_ylabel("Confidence")
        ax[i].set_xticks(x)
        ax[i].set_xticklabels(samples_label)
        ax[i].set_title(name)
        ax[i].grid(axis="y", linestyle="--", alpha=0.5)

    handles, labels = ax[0].get_legend_handles_labels()

    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=8)
    fig.suptitle("Confidence over each class vs Mean Confidence", y=1.1)

    fig.tight_layout()
    fig.savefig("./figures/problem3/test_confidence.pdf", bbox_inches="tight")


def plot_class_balance_test():
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    rda = np.load("./models/1d/RDA_7680preded_class.npy")
    knn = np.load("./models/1d/KNN_7680preded_class.npy")
    names = ["KNN", "RDA"]
    unique_labels = np.unique(rda)
    class_dist = np.zeros((2, 7))
    model_data = [knn, rda]
    for model_idx, data in enumerate(model_data):
        for class_idx, cls in enumerate(unique_labels):
            class_dist[model_idx, class_idx] = np.sum(data == cls) / data.shape[0]

        class_labels = "1", "2", "3", "4", "5", "6", "7"
        ax[model_idx].pie(class_dist[model_idx], labels=class_labels, autopct="%1.1f%%")
        ax[model_idx].set_title(names[model_idx])
    fig.suptitle("Class balances for 7680 training samples")
    fig.tight_layout()
    fig.savefig("./figures/problem3/class_balance_test.pdf")


def main():
    plot_confidence_acc()
    plot_confidence_over_sample_diff()
    plot_confidence_for_test()
    plot_class_balance_test()


if __name__ == "__main__":
    main()
