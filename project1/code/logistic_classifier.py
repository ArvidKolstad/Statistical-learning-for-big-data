import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from dim_red import dimension_reduction
from knn_classifier import classifier_preformance
from sklearn.preprocessing import StandardScaler

def tune_dim_red(x_train, y_train, n_dims, n_folds):
        # cv_scores = []
        results = {} # Ändrat

        for n_dim in n_dims:
            x_reduced, _ = dimension_reduction(x_train, n_dim_pca=n_dim)

            model = LogisticRegression(solver='lbfgs', max_iter=1000)

            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x_reduced)

            scores = cross_val_score(
                model, 
                x_scaled, 
                y_train, 
                cv=n_folds, 
                scoring="accuracy", 
                n_jobs=-1)
                
            # cv_scores.append(scores.mean())
            results[n_dim] = scores.mean() # Ändrat
            # print(f"n_dim={n_dim}: {scores.mean():.4f}")

        # #plot
        # plt.figure()
        # plt.plot(n_dims, cv_scores)
        # plt.xlabel('n_dim')
        # plt.ylabel('CV Accuracy')
        # plt.grid()

        # find best n_dim
        # best_n_dim = n_dims[np.argmax(cv_scores)]
        best_n_dim = max(results, key=results.get) # Ändrat 
        # print(f'Optimal dim. reduction: {best_n_dim}')

        # bästa dim nytt
        x_best, _ = dimension_reduction(x_train, n_dim_pca=best_n_dim)

        # träna nytt
        scaler = StandardScaler()
        x_best_scaled = scaler.fit_transform(x_best)

        best_model = LogisticRegression(solver='lbfgs', max_iter=1000)
        best_model.fit(x_best_scaled, y_train)


        return best_model, best_n_dim, scaler

def main():
    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    test_labels = np.load("./data/test_labels.npy")
    test_matrix = np.load("./data/test_matrix.npy")

    n_dim = tune_dim_red(training_matrix, training_labels, range(10,151,10), 10)

    training_matrix, test_matrix = dimension_reduction(
        training_matrix, test_data=test_matrix, n_dim_pca=n_dim)

    model = LogisticRegression(solver='lbfgs', max_iter=1000)
    model.fit(training_matrix, training_labels)

    classifier_preformance(model, test_matrix, test_labels)


if __name__ == "__main__":
    main()
