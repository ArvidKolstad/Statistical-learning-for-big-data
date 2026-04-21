import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from dim_red import dimension_reduction

def classifier_preformance(classifier, x_test, y_test):
    digits = range(0,10)
    y_pred = classifier.predict(x_test)
    
    cm = confusion_matrix(y_test, y_pred, labels=digits)
    ConfusionMatrixDisplay(cm, display_labels=digits).plot(cmap='Greens')

    accuracy = accuracy_score(y_pred, y_test)
    errors = (y_pred != y_test).sum()
    print(f'Model accuracy: {accuracy}')
    print(f'No. model errrors: {errors}')
    plt.show()

def tune_knn(x_train, y_train, k_values, n_folds):
    cv_scores = []
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train, cv=n_folds, scoring="accuracy")
        cv_scores.append(scores.mean())

    #plot
    plt.figure()
    plt.plot(k_values, cv_scores)
    plt.xlabel('k')
    plt.ylabel('CV Accuracy')
    plt.grid()

    # find best k
    best_k = k_values[np.argmax(cv_scores)]
    print(f'Best k value: {best_k}')

    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x_train, y_train)
    return best_knn

def tune_knn_and_dim_red(x_train, y_train, k_values, n_folds, n_dims):
    results = {}  # (n_dim, k) -> cv_score

    for n_dim in n_dims:
        x_reduced, _ = dimension_reduction(x_train, n_dim_pca=n_dim)
        
        for k in k_values:
            knn = KNeighborsClassifier(n_neighbors=k)
            scores = cross_val_score(knn, x_reduced, y_train, cv=n_folds, scoring="accuracy")
            results[(n_dim, k)] = scores.mean()
            # print(f"n_dim={n_dim}, k={k}: {scores.mean():.4f}")

    # find best combination
    best_dim, best_k = max(results, key=results.get)
    # print(f"\nBest n_dim={best_dim}, best k={best_k}, accuracy={results[(best_dim, best_k)]:.4f}")

    # plot heatmap
    dims_list = list(n_dims)
    k_list = list(k_values)
    grid = np.array([[results[(d, k)] for k in k_list] for d in dims_list])

    # plt.figure(figsize=(10, 6))
    # plt.imshow(grid, aspect="auto", cmap="viridis",
    #            extent=[k_list[0], k_list[-1], dims_list[-1], dims_list[0]])
    # plt.colorbar(label="CV Accuracy")
    # plt.xlabel("k")
    # plt.ylabel("n_dim")
    # plt.title("kNN accuracy over k and PCA dimensions")
    # plt.show()

    # refit best model
    x_best, _ = dimension_reduction(x_train, n_dim_pca=best_dim)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(x_best, y_train)

    return best_knn, best_dim

def main():
    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    test_labels = np.load("./data/test_labels.npy")
    test_matrix = np.load("./data/test_matrix.npy")

    knn, n_dim = tune_knn_and_dim_red(training_matrix, training_labels, range(1,10), 10, range(10,101,10))

    training_matrix, test_matrix = dimension_reduction(
        training_matrix, test_data=test_matrix, n_dim_pca=n_dim)
    
    classifier_preformance(knn, test_matrix, test_labels)

if __name__ == "__main__":
    main()
