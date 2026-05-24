import numpy as np
from torch.utils.data import Dataset
import pandas as pd


class AnimalPictures(Dataset):
    def __init__(self, in_features, labels):
        self.in_features = in_features
        self.labels = labels

    def __len__(self):
        return self.in_features.shape[0]

    def __getitem__(self, idx):
        x = self.in_features[idx]
        y = self.labels[idx]

        return x, y


def split_data():
    keep_data_frac = 0.05
    PATHIM = "data/cnd_large/images.csv"
    PATHLB = "data/cnd_large/labels.csv"

    images = pd.read_csv(PATHIM, sep=",", index_col=0)
    labels = pd.read_csv(PATHLB, sep=",", index_col=0)

    labels = labels.rename(columns={"0": "label"})

    df = images.join(labels)

    unique_labels = df["label"].unique()

    df_train = pd.DataFrame()
    for label in unique_labels:
        df_label = df[df["label"] == label]
        df_label_train = df_label.sample(frac=keep_data_frac)
        df_train = pd.concat([df_train, df_label_train])
    df_train = df_train

    df_train = df.sample(frac=1).reset_index(drop=True)
    train_labels = df_train["label"].to_numpy()
    train_matrix = df_train.drop(columns="label").to_numpy()
    """
    np.save("./data/train_labels", train_labels)
    np.save("./data/train_matrix", train_matrix)
    """
    df_train.to_csv("data_500.csv")


def main():
    labels = np.load("./data/train_labels.npy")
    matrix = np.load("./data/train_matrix.npy")
    df = pd.DataFrame({"label": labels, "matrix": matrix})

    df_train.to_csv("df")


if __name__ == "__main__":
    main()
