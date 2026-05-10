from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def lasso_embedding(images, labels, C=1.0, return_info=False):
    # lasso = LogisticRegression(l1_ratio=1, C=C, solver="liblinear", max_iter=1000)
    lasso = LogisticRegression(penalty="l1",C=C,solver="liblinear",max_iter=1000)
    lasso.fit(images, labels)

    selected_mask = lasso.coef_[0] != 0
    filtered_images = images[:, selected_mask]

    if return_info:
        return filtered_images, lasso.coef_[0], selected_mask
    else:
        return filtered_images


def main():
    training_matrix = np.load("./data/train_matrix.npy")
    training_labels = np.load("./data/train_labels.npy")

    X_reduced, coefs, selected_mask = lasso_embedding(
        training_matrix, training_labels, C=0.001, return_info=True
    )
    print(f"Selected {X_reduced.shape[1]} pixels out of {training_matrix.shape[1]}")

    plt.figure()
    plt.imshow(selected_mask.reshape(64, 64), cmap="hot")
    plt.title("Choosen pixels")

    plt.figure()
    vmax = np.abs(coefs).max()
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    plt.imshow(coefs.reshape(64, 64), cmap="RdBu_r", norm=norm)
    plt.colorbar()
    plt.title("Coeffient per pixel")
    plt.savefig("../figures/dim_filter/lasso.png")
    plt.show()


if __name__ == "__main__":
    main()
