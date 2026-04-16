import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline


def dimension_reduction(
    train_data, train_label=None, n_dimensions=2, plot=False, save_path=None
):

    pca_tsne = Pipeline(
        [
            ("pca", PCA(n_components=0.95)),
            ("tsne", TSNE(n_components=n_dimensions)),
        ]
    )

    train_reduced = pca_tsne.fit_transform(train_data)

    if plot:
        if n_dimensions == 2:
            plt.figure(figsize=(12, 8))
            plt.scatter(
                train_reduced[:, 0], train_reduced[:, 1], c=train_label, cmap="jet"
            )
            plt.colorbar()
            plt.axis("off")

        elif n_dimensions == 3:
            fig = plt.figure(figsize=(10, 10))
            ax = plt.axes(projection="3d")

            sc = ax.scatter(
                train_reduced[:, 0],
                train_reduced[:, 1],
                train_reduced[:, 2],
                c=train_label,
                cmap="jet",
            )

            fig.colorbar(sc)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    return train_reduced


def main():
    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    dimensions = 3
    save_fig = "../figures/dim_reduced_data.png"

    dimension_reduction(
        training_matrix,
        train_label=training_labels,
        n_dimensions=dimensions,
        plot=True,
        save_path=save_fig,
    )


if __name__ == "__main__":
    main()

