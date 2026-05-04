import numpy as np
import matplotlib.pyplot as plt

def flip_data(images_path, labels_path, flip_frac, save_images = False):
    images = np.load(images_path)
    labels = np.load(labels_path)
    out = images.copy()

    for label in [0,1]:
        indices = np.where(labels == label)[0]
        n_replace = int(len(indices) * flip_frac)

        chosen = np.random.choice(indices, size=n_replace, replace=False)
        to_flip = out[chosen]
        to_flip = to_flip.reshape(-1,64,64) # reshape
        to_flip = to_flip[:, ::-1, :] # flip
        to_flip = to_flip.reshape(-1,64*64) # flatten
        out[chosen] = to_flip

    if save_images:
        np.save(images_path[:-4] + f"_{flip_frac}_flipped", out)

    return out

def main():
    flip_data("./data/train_matrix.npy", "./data/train_labels.npy", 0.5, True)
    flip_data("./data/test_matrix.npy", "./data/test_labels.npy", 0.5, True)

if __name__ == "__main__":
    main()