import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def dimension_reduction(
    train_data,
    test_data=None,
    train_label=None,
    tsne=False,
    n_dim=2,
    n_dim_pca=0.95,
    plot=False,
    save_path=None,
):

    pca = PCA(n_components=n_dim_pca)
    train_pca = pca.fit_transform(train_data)

    if tsne:
        # tsne = TSNE(n_components=n_dimensions, random_state=42)
        tsne = TSNE(
            n_components=n_dim,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=42,
        )
        train_reduced = tsne.fit_transform(train_pca)
        test_reduced = None

    else:
        train_reduced = train_pca

        if test_data is not None:
            test_reduced = pca.transform(test_data)
        else:
            test_reduced = None

    if plot:
        if save_path:
            suffix = "tsne" if tsne else "pca"

        if train_label is None:
            raise ValueError("train_label must be provided for plotting")

        if n_dim == 2:
            plt.figure(figsize=(12, 8))
            plt.scatter(
                train_reduced[:, 0], train_reduced[:, 1], c=train_label, cmap="jet"
            )
            plt.colorbar()
            plt.axis("off")
            plt.title("2D projection")

            if save_path:
                plt.savefig(
                    save_path.replace(".png", f"_{suffix}_2d.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            plt.show()

        if n_dim == 3:
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
            ax.set_title("3D projection")

            if save_path:
                plt.savefig(
                    save_path.replace(".png", f"_{suffix}_3d.png"),
                    dpi=300,
                    bbox_inches="tight",
                )

            plt.show()

    return train_reduced, test_reduced


def main():
    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")

    dimensions = 3
    save_fig = "../figures/dim_reduced_data.png"

    dimension_reduction(
        training_matrix,
        train_label=training_labels,
        tsne=True,
        n_dim=dimensions,
        plot=True,
        save_path=save_fig,
    )


if __name__ == "__main__":
    main()
