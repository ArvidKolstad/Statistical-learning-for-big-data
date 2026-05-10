import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif

# ANOVA F-test filtering to k top pixels 
def f_score_filter(images, labels, k, return_scores = False):
    filter = SelectKBest(f_classif, k=k)
    filter.fit(images, labels)
    filterd_images = filter.transform(images)

    if return_scores: # For visualiztion
        selected_pixel_idxs = filter.get_support(indices=True)
        scores = filter.scores_ 

        return filterd_images, scores, selected_pixel_idxs
    else:
        return filterd_images

def main():
    training_matrix = np.load("./data/train_matrix.npy")
    training_labels = np.load("./data/train_labels.npy")

    filterd_images, scores, selected_pixel_idxs = f_score_filter(training_matrix, training_labels, 163, True)

    print(training_matrix.shape)
    print(filterd_images.shape)

    pixels = scores.copy()
    pixels[selected_pixel_idxs] += 10000
    plt.figure()
    plt.imshow(pixels.reshape(64, 64), cmap="hot")
    plt.title("Choosen pixels")

    plt.figure()
    plt.imshow(scores.reshape(64, 64), cmap="hot")
    plt.colorbar()
    plt.title("F-score per pixel")
    plt.show()


if __name__ == "__main__":
    main()