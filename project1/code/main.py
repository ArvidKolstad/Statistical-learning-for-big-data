import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit


def split_load_data():

    unique_labels =df["label"].unique())

     
    PATHIM = "data/images.csv"
    PATHLB = "data/labels.csv"

    images = pd.read_csv(PATHIM, sep=",", index_col=0)
    labels = pd.read_csv(PATHLB, sep=",", index_col=0)

    labels = labels.rename(columns={"0": "label"})

    # print(f"Shape images (before): {images.shape}")
    # print(f"Shape labels (before): {labels.shape}")

    df = images.join(labels)
    df["mean_intensity"] = images.mean(axis=1)

    ax = df.pivot(columns="label", values="mean_intensity").hist(bins=40)
    fig = ax[0, 0].get_figure()
    fig.savefig("../figures/data_histogram.png")
    # print(df["label"].unique())
    images = images.to_numpy()
    labels = labels.to_numpy()
    return images, labels


def preprocess_data(images, labels, data_splits, test_size, train_size):
    print(images.shape)
    sp = ShuffleSplit(n_splits=data_splits, test_size=test_size, train_size=train_size)
    train = sp.split(images, labels)
    print(train)
    for i, (train, test) in enumerate(sp.split(images, labels)):
        print("Train_size batch", train.shape)
        print("Test batch", test.shape)

    ##print(test.shape)


def main():
    n_splits = 10
    train_size = 0.5
    test_size = 0.1

    images, labels = load_data()
    preprocess_data(images, labels, n_splits, test_size, train_size)


if __name__ == "__main__":
    main()
