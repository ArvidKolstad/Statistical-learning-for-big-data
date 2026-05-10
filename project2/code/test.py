import numpy as np
import matplotlib.pyplot as plt
from itertools import product

from logistic_regression import LogisticRegressionModel
from train_utils import kCV
from f_test_filter_selection import f_score_filter
from reg_disc import RegularizedDiscriminantAnalysis, run_RDA_training

# ─────────────────────────────────────────────
#  RDA wrapper — applies f-filter inside each
#  fold to avoid leaking val data into selection
# ─────────────────────────────────────────────


class RDAWithFilter:
    """
    Wraps RDA so that f-score filtering is applied inside train(),
    meaning the selector is only ever fit on the training split.

    kCV passes raw numpy arrays as tuples when data_loaders=False:
        train_data = (X_train, y_train)
        val_data   = (X_val,   y_val)
    """

    def __init__(self, k_features, classes, lmbda, gamma, class_names):
        self.k_features = k_features
        self.rda_settings = {
            "in_features": k_features,
            "classes": classes,
            "lmbda": lmbda,
            "gamma": gamma,
            "class_names": class_names,
        }

    def train(self, train_data, val_data):
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Fit filter on train, get selected indices
        X_train_f, _, selected_idx = f_score_filter(
            X_train, y_train, self.k_features, return_scores=True
        )
        # Apply same indices to val — no fitting on val data
        X_val_f = X_val[:, selected_idx]test

        self.rda = RegularizedDiscriminantAnalysis(**self.rda_settings)
        score = run_RDA_training(
            self.rda,
            X_train_f,
            y_train,
            X_val_f,
            y_val,
        )
        return score


class LogRegWithFilter:
    def __init__(self, k_features):
        self.k_features = k_features

    def train(self, train_data, val_data):
        X_train, y_train = train_data
        X_val, y_val = val_data

        # Fit filter on train, get selected indices
        X_train_f, _, selected_idx = f_score_filter(
            X_train, y_train, self.k_features, return_scores=True
        )
        # Apply same indices to val — no fitting on val data
        X_val_f = X_val[:, selected_idx]

        self.log_reg = LogisticRegressionModel()
        self.log_reg.train(X_train_f, y_train)
        score = self.log_reg.score(X_train, y_train)
        return score

    def save(self, file_name):
        self.log_reg.save(file_name)

    def load(self, file_name):
        self.log_reg = LogisticRegressionModel()
        self.log_reg.load(file_name)
        return self.log_reg


# ─────────────────────────────────────────────
#  Parameter sweep
# ─────────────────────────────────────────────


def param_sweep(
    train_matrix, train_labels, k_folds, k_features_list, lmbda_list, gamma_list
):
    results = []
    total = len(k_features_list) * len(lmbda_list) * len(gamma_list)

    for i, (k_feat, lmbda, gamma) in enumerate(
        product(k_features_list, lmbda_list, gamma_list), 1
    ):
        model_settings = {
            "k_features": k_feat,
            "classes": 2,
            "lmbda": lmbda,
            "gamma": gamma,
            "class_names": ["Cats", "Dogs"],
        }
        training_settings = {
            "train_data": None,
            "val_data": None,
        }

        fold_scores = kCV(
            k=k_folds,
            model_class=RDAWithFilter,
            train_input=train_matrix,
            train_target=train_labels,
            model_settings=model_settings,
            training_settings=training_settings,
            data_loaders=False,
        )

        mean_acc = np.mean(fold_scores)
        print(
            f"[{i}/{total}]  k={k_feat}  λ={lmbda}  γ={gamma}  →  mean={mean_acc:.4f}"
        )

        results.append(
            {
                "k": k_feat,
                "lmbda": lmbda,
                "gamma": gamma,
                "mean": mean_acc,
                "folds": fold_scores,
            }
        )

    results.sort(key=lambda r: r["mean"], reverse=True)
    return results


# ─────────────────────────────────────────────
#  Plotting — one subplot per parameter,
#  averaging over the other two each time
# ─────────────────────────────────────────────


def plot_sweep_results(results, save_path="../figures/RDA/sweep_results.png"):
    """
    3 subplots showing mean accuracy vs each parameter independently,
    averaging over all values of the other two parameters.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # --- k_features ---
    k_vals = sorted(set(r["k"] for r in results))
    k_means = [np.mean([r["mean"] for r in results if r["k"] == k]) for k in k_vals]
    k_std = [np.std([r["mean"] for r in results if r["k"] == k]) for k in k_vals]

    axes[0].errorbar(k_vals, k_means, k_std, marker="o")
    axes[0].set_xlabel("k features")
    axes[0].set_ylabel("Mean accuracy")
    axes[0].set_title("Effect of k features")
    axes[0].grid(alpha=0.4)
    axes[0].set_xscale("log")

    # --- lmbda ---
    l_vals = sorted(set(r["lmbda"] for r in results))
    l_means = [np.mean([r["mean"] for r in results if r["lmbda"] == l]) for l in l_vals]
    l_std = [np.std([r["mean"] for r in results if r["lmbda"] == l]) for l in l_vals]

    axes[1].errorbar(l_vals, l_means, yerr=l_std, marker="o", color="orange")
    axes[1].set_xlabel("λ (0=QDA, 1=LDA)")
    axes[1].set_title("Effect of λ")
    axes[1].grid(alpha=0.4)

    # --- gamma ---
    g_vals = sorted(set(r["gamma"] for r in results))
    g_means = [np.mean([r["mean"] for r in results if r["gamma"] == g]) for g in g_vals]
    g_std = [np.std([r["mean"] for r in results if r["gamma"] == g]) for g in g_vals]

    axes[2].errorbar(g_vals, g_means, yerr=g_std, marker="o", color="green")
    axes[2].set_xlabel("γ (covariance shrinkage)")
    axes[2].set_title("Effect of γ")
    axes[2].grid(alpha=0.4)

    fig.suptitle("RDA + F-filter parameter sweep", fontsize=13)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────


def main():
    train_matrix = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    k_features_list = [100, 300, 500, 700, 900, 1000, 1500, 2000, 2500]
    lmbda_list = [0.0, 0.25, 0.5, 0.75, 1.0]
    gamma_list = [0.0, 0.2, 0.4, 0.5]

    results = param_sweep(
        train_matrix,
        train_labels,
        k_folds=5,
        k_features_list=k_features_list,
        lmbda_list=lmbda_list,
        gamma_list=gamma_list,
    )

    print("\n── Top 5 configurations ──")
    for r in results[:5]:
        print(f"  k={r['k']:>5}  λ={r['lmbda']}  γ={r['gamma']}  acc={r['mean']:.4f}")

    plot_sweep_results(results)


if __name__ == "__main__":
    main()
