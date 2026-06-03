import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from torch.utils.data import Dataset


class B11_dataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.labels[index]
        return x, y


def get_processed_test():
    df_input = pd.read_csv("./data/X_TE.csv")
    index = [
        7,
        37,
        100,
        141,
        158,
        179,
        210,
        228,
        276,
        308,
        427,
        519,
        550,
        558,
        597,
        752,
        762,
        796,
        809,
    ]

    df_input = df_input.iloc[:, index]
    np.save("./data/test_set", df_input)


def visualize_dataset():
    df_input = pd.read_csv("./data/X_TR.csv")
    df_labels = pd.read_csv("./data/y_TR.csv")
    df_labels = df_labels - 1

    unique_labels = np.unique(df_labels.to_numpy())
    df_total = pd.concat([df_input, df_labels], axis=1)

    pca = PCA(
        n_components=820,
        copy=True,
        whiten=False,
        svd_solver="full",
    )
    inputs = df_input.to_numpy()
    labels = df_labels.to_numpy()

    pca.fit(inputs)
    cum_variance = pca.explained_variance_ratio_.cumsum()
    fig, ax = plt.subplots()
    ax.plot(cum_variance)
    ax.set_xlabel("PC")
    ax.set_ylabel("Cumulative variance ratio")
    fig.savefig("./figures/vis_data/cumulative_variance_ratio_pca.pdf")

    class_dist = np.zeros(np.max(unique_labels) + 1)

    for idx, cls in enumerate(unique_labels):
        class_dist[idx] = np.sum(df_total["class"] == cls) / df_total.shape[0]

    fig, ax = plt.subplots()
    class_labels = "1", "2", "3", "4", "5", "6", "7"
    ax.pie(class_dist, labels=class_labels, autopct="%1.1f%%")
    ax.set_title("Class balance")
    fig.tight_layout()
    fig.savefig("./figures/vis_data/class_balance.pdf")
    print(f"Mean value: {np.mean(df_input.mean()):.3f}")
    print(f"STD value: {np.mean(df_input.std()):.3f}")
    print(f"Max value: {np.max(df_input.max()):.3f}")
    print(f"Min value: {np.min(df_input.min()):.3f}")
    print(f"Total samples: {df_input.shape[0]}")
    print(f"Input dim: {df_input.shape[1]}")
    print(f"Unique Classes: {unique_labels}")

    f_statistics, p_statistics = f_regression(inputs, labels)
    saved_features_index = []

    for idx, f_val in enumerate(f_statistics):
        if f_val >= 10:
            saved_features_index.append(idx)
    print(f"saved_features: {len(saved_features_index)}")

    fig, ax1 = plt.subplots(figsize=(14, 5))

    x = np.arange(len(saved_features_index))
    width = 0.30
    gap = 0.05

    ax2 = ax1.twinx()

    bars1 = ax1.bar(
        x - width / 2 - gap / 2,
        f_statistics[saved_features_index],
        width,
        color="steelblue",
        alpha=0.8,
        label="F-value",
    )
    bars2 = ax2.bar(
        x + width / 2 + gap / 2,
        p_statistics[saved_features_index],
        width,
        color="tomato",
        alpha=0.8,
        label="P-value",
    )
    ax1.bar_label(bars1, fmt="%.1f", padding=3, fontsize=7, color="steelblue")
    ax2.bar_label(bars2, fmt="%.2e", padding=3, fontsize=7, color="tomato")

    ax1.set_xlabel("Feature index")
    ax1.set_ylabel("F-value", color="steelblue")
    ax2.set_ylabel("P-value", color="tomato")
    ax1.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="tomato")
    ax2.set_yscale("log")

    ax1.set_xticks(x)
    ax1.set_xticklabels(saved_features_index, rotation=45, ha="right", fontsize=8)

    lines = [bars1, bars2]
    labels = ["F-value", "P-value"]
    ax1.legend(lines, labels, loc="upper left")

    fig.suptitle("F-test filtering")
    fig.tight_layout()
    fig.savefig("./figures/vis_data/f_test.pdf")


def get_data_different_sample_sizes():
    index = [
        7,
        37,
        100,
        141,
        158,
        179,
        210,
        228,
        276,
        308,
        427,
        519,
        550,
        558,
        597,
        752,
        762,
        796,
        809,
    ]

    df_input = pd.read_csv("./data/X_TR.csv")
    df_labels = pd.read_csv("./data/y_TR.csv")
    df_labels = df_labels - 1
    df_input = df_input.iloc[:, index]

    df_big = pd.concat([df_input, df_labels], axis=1)
    df_big = df_big.sample(frac=1, random_state=42).reset_index(drop=True)

    unique_classes = df_big["class"].unique()
    sizes = [200, 500, 1000, 3000, 5000, len(df_big)]
    splits = {}

    for size in sizes:
        chunks = []
        for cls in unique_classes:
            cls_df = df_big[df_big["class"] == cls]
            n = round(size * len(cls_df) / len(df_big))
            chunks.append(cls_df.iloc[:n])

        df_split = (
            pd.concat(chunks).sample(frac=1, random_state=42).reset_index(drop=True)
        )

        splits[size] = df_split

    for size, subset in splits.items():
        print(size)
        print(subset["class"].value_counts(normalize=True).round(3))
        np.save(f"./data/{size}_labels", subset["class"].to_numpy())
        np.save(f"./data/{size}_input", subset.drop(columns="class").to_numpy())


def get_random_labels():
    df_labels = pd.read_csv("./data/y_TR.csv")
    df_labels = df_labels - 1
    df_labels = df_labels.sample(frac=1).reset_index(drop=True)
    np.save("./data/7680_labels_random", df_labels["class"].to_numpy())


def plot_variance_between_classes():
    df_input = pd.read_csv("./data/X_TR.csv")
    df_labels = pd.read_csv("./data/y_TR.csv")
    index = [
        7,
        37,
        100,
        141,
        158,
        179,
        210,
        228,
        276,
        308,
        427,
        519,
        550,
        558,
        597,
        752,
        762,
        796,
        809,
    ]

    index_labels = [
        "7",
        "37",
        "100",
        "141",
        "158",
        "179",
        "210",
        "228",
        "276",
        "308",
        "427",
        "519",
        "550",
        "558",
        "597",
        "752",
        "762",
        "796",
        "809",
    ]
    df_input = df_input.iloc[:, index]

    unique_labels = np.unique(df_labels.to_numpy())
    class_variance_feature = []
    for cls in unique_labels:
        inputs = df_input[df_labels["class"] == cls].to_numpy()
        class_variance_feature.append(np.std(inputs, axis=0))
    class_variance_feature = np.vstack(class_variance_feature)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(class_variance_feature)
    for i in range(len(unique_labels)):
        for j in range(len(index)):
            ax.text(
                j,
                i,
                class_variance_feature[i, j].round(2),
                ha="center",
                va="center",
                color="w",
            )
    ax.set_xticks(range(len(index_labels)), labels=index_labels)
    ax.set_yticks(range(len(unique_labels)), labels=unique_labels)

    ax.set_xlabel("Feature index")
    ax.set_ylabel("Class")

    ax.set_title("Variance of features between classes")
    fig.tight_layout()
    fig.savefig("./figures/vis_data/variance_of_classes.pdf")


def plot_covar_matrix():
    inputs = np.load("./data/7680_input.npy")
    cov_matrix = np.corrcoef(inputs, rowvar=False)
    index = [
        "7",
        "37",
        "100",
        "141",
        "158",
        "179",
        "210",
        "228",
        "276",
        "308",
        "427",
        "519",
        "550",
        "558",
        "597",
        "752",
        "762",
        "796",
        "809",
    ]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(cov_matrix)
    for i in range(len(index)):
        for j in range(len(index)):
            ax.text(
                j, i, cov_matrix[i, j].round(2), ha="center", va="center", color="w"
            )
    ax.set_xticks(range(len(index)), labels=index)
    ax.set_yticks(range(len(index)), labels=index)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Feature Index")

    ax.set_title("Covariance Matrix")
    fig.tight_layout()
    fig.savefig("./figures/vis_data/covariance_matrix.pdf")


def get_data_balance(labels):
    unique_labels = np.unique(labels)
    balance = []
    total_length = labels.shape[0]
    for cls in unique_labels:
        balance.append(1 / (np.sum(labels == cls) / total_length))
    return np.array(balance)


def main():
    # visualize_dataset()
    # get_data_different_sample_sizes()
    # plot_covar_matrix()
    # get_random_labels()
    # plot_variance_between_classes()
    get_processed_test()


if __name__ == "__main__":
    main()
