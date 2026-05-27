from baycomp import two_on_single
import numpy as np


def get_p_scores(models_paths, performance_score, runs, rope=0.01):

    index = get_performance_idx(performance_score)
    matrix = np.zeros((4, 4))

    for idx1, path1 in enumerate(models_paths):
        model_1 = np.load("models/" + path1)[:, index]
        print(path1)
        print(model_1)
        for idx2, path2 in enumerate(models_paths):
            model_2 = np.load("models/" + path2)[:, index]
            p_left, p_rope, p_right = two_on_single(
                model_1, model_2, rope=rope, runs=runs
            )
            matrix[idx1, idx2] = p_rope
    return matrix


def get_performance_idx(performance_score):
    if performance_score == "accuracy":
        index = 0
    elif performance_score == "f1-score":
        index = 1
    elif performance_score == "AOC":
        index = 2
    else:
        raise NotImplementedError
    return index


def get_rope_size(model_paths, performance_score, runs):
    ropes = np.arange(0.0001, 0.1, 0.001)
    index = get_performance_idx(performance_score)
    for rope in ropes:
        saved_p_ropes = []
        for path1 in model_paths:
            model_1 = np.load("models/" + path1)[:, index]
            for path2 in model_paths:
                model_2 = np.load("models/" + path2)[:, index]
                _, p_rope, _ = two_on_single(model_1, model_2, rope=rope, runs=runs)
                saved_p_ropes.append(p_rope)
        mean_p_ropes = np.mean(saved_p_ropes)
        if mean_p_ropes > 0.99:
            best_rope = rope
            print(f"Good rope was found: {best_rope}")
            return best_rope

    best_rope = 0.1
    print("Good rope wasn't found")
    return best_rope


def main():
    samples = 1000
    runs = 10
    rope = 0.00000001
    models_paths = [
        f"KNN_{samples}_random.npy",
        f"LogReg_{samples}_random.npy",
        f"XGB_{samples}_random.npy",
        f"RDA_{samples}_random.npy",
    ]
    performance_score = "accuracy"
    matrix = get_p_scores(models_paths, performance_score, runs, rope=rope)
    print(matrix)


if __name__ == "__main__":
    main()
