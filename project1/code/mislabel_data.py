import numpy as np

def mislabel_data(labels, mislabel_frac, save_labels = False):
    out = labels.copy()

    for digit in range(10):
        indices = np.where(labels == digit)[0]
        n_replace = int(len(indices) * mislabel_frac)

        chosen = np.random.choice(indices, size=n_replace, replace=False)
        other_digits = np.array([d for d in range(10) if d != digit])
        out[chosen] = np.random.choice(other_digits, size=n_replace)

    if save_labels:
        np.save(f"./data/train_labels_{mislabel_frac}_mislabel", out)

    return out

def verify_mislabel(original_labels, changed_labels):
    print("No. mislabeld datapoints")
    original = len(original_labels)
    changed = np.sum((original_labels != changed_labels))
    print(f"Total: {changed}/{original} ({changed/original:.1%})")

    for digit in range(10):
        original = np.sum(original_labels == digit)
        changed  = np.sum((original_labels == digit) & (original_labels != changed_labels))
        print(f"Digit {digit}: {changed}/{original} ({changed/original:.1%})")

def main():
    training_labels = np.load("./data/train_labels.npy")
    training_labels_mislabeled = mislabel_data(training_labels, 0.5, True)
    verify_mislabel(training_labels, training_labels_mislabeled)

if __name__ == "__main__":
    main()