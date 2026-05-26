from baycomp import two_on_single
import numpy as np


def get_p_scores(models_paths, performance_score, runs):
    if performance_score == "accuracy":
        index = 0
    elif performance_score == "f1-score":
        index = 1
    elif performance_score == "AOC":
        index = 2
    else:
        raise NotImplementedError

    matrix = np.zeros((4, 4))
    for idx1, path1 in enumerate(models_paths):
        model_1 = np.load("models/" + path1)[:, index]
        print(path1)
        print(model_1)
        for idx2, path2 in enumerate(models_paths):
            model_2 = np.load("models/" + path2)[:, index]
            print(path2)
            print(model_2)
            p_left, _ = two_on_single(model_1, model_2, runs=runs)
            matrix[idx1, idx2] = p_left
    return matrix


def main():
    print("Hello")
    samples = 1000
    runs = 10
    models_paths = [
        f"LogReg_{samples}.npy",
        f"KNN_{samples}.npy",
        f"XGB_{samples}.npy",
        f"RDA_{samples}.npy",
    ]
    performance_score = "f1-score"
    matrix = get_p_scores(models_paths, performance_score, runs)
    print(matrix)


if __name__ == "__main__":
    main()
