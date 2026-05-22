from torch.utils.data import Dataset


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
