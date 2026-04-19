import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit


def split_data():
    test_data_frac = 0.1
    PATHIM = "data/images.csv"
    PATHLB = "data/labels.csv"

    images = pd.read_csv(PATHIM, sep=",", index_col=0)
    labels = pd.read_csv(PATHLB, sep=",", index_col=0)

    labels = labels.rename(columns={"0": "label"})

    # print(f"Shape images (before): {images.shape}")
    # print(f"Shape labels (before): {labels.shape}")

    df = images.join(labels)

    unique_labels = df["label"].unique()
    df_test = pd.DataFrame()
    df_train = pd.DataFrame()

    for label in unique_labels:
        df_label = df[df["label"] == label]
        df_label_test = df_label.sample(frac=test_data_frac)
        df_label_train = df_label.drop(df_label_test.index)

        df_test = pd.concat([df_test, df_label_test])
        df_train = pd.concat([df_train, df_label_train])

    test_res = df_test.sample(frac=1).reset_index(drop=True)
    train_res = df_train.sample(frac=1).reset_index(drop=True)

    train_labels = train_res["label"].to_numpy()
    test_labels = test_res["label"].to_numpy()

    train_matrix = train_res.drop(columns="label").to_numpy()
    test_matrix = test_res.drop(columns="label").to_numpy()

    np.save("./data/train_labels", train_labels)
    np.save("./data/train_matrix", train_matrix)
    np.save("./data/test_labels", test_labels)
    np.save("./data/test_matrix", test_matrix)

    test_res.to_csv("./data/test_data.csv")
    train_res.to_csv("./data/train_data.csv")


def preprocess_data(images, labels, data_splits, test_size, train_size):
    print(images.shape)
    sp = ShuffleSplit(n_splits=data_splits, test_size=test_size, train_size=train_size)
    train = sp.split(images, labels)
    print(train)
    for i, (train, test) in enumerate(sp.split(images, labels)):
        print("Train_size batch", train.shape)
        print("Test batch", test.shape)


def main():
    split_data()


if __name__ == "__main__":
    main()
