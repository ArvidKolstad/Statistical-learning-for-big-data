import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from f_test_filter_selection import f_score_filter
from lasso_regression_selection import lasso_embedding


# Wrapper class
class LogisticRegressionModel:
    def __init__(self, **settings):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=42 # Bör kanske ändras sen för repetering
            ))
        ])

    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, Y):
        return self.model.score(X, Y)

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)
        return self



# Train model
def train_logistic_regression(
    x_train, 
    y_train, 
    n_folds=10, 
    save_model=None):

    model = LogisticRegressionModel()

    print('CV started...')
    scores = cross_val_score(
        model.model,
        x_train,
        y_train,
        cv=n_folds,
        scoring='accuracy',
        n_jobs=-1)
    print('CV finished.')

    model.fit(x_train, y_train)

    if save_model:
        model.save(save_model)

    return scores.mean()


# Use CV to find the best number of features for f test
def find_best_k_f_test(
    x_train,
    y_train,
    k_values,
    n_folds=10):

    results = {}

    for k in k_values:

        print(f'Testing k = {k}')

        # Feature selection
        x_train_reduced = f_score_filter(
            x_train,
            y_train,
            k=k)

        # Logistic regression
        model = LogisticRegressionModel()

        # FInd CV score
        scores = cross_val_score(
            model.model,
            x_train_reduced,
            y_train,
            cv=n_folds,
            scoring='accuracy',
            n_jobs=-1)

        mean_score = scores.mean()
        results[k] = mean_score

        print(f'k = {k}, CV accuracy = {mean_score:.4f}')

    best_k = max(results, key = results.get)

    print(f"\nBest k found: {best_k}")
    print(f"Best CV accuracy: {results[best_k]:.4f}")

    return best_k, results


# Use CV to find best number of features for Lasso
def find_best_C_lasso(
    x_train,
    y_train,
    C_values,
    n_folds=10):

    results = {}

    for C in C_values:
        print(f'Testing C = {C}')

        scores = []

        from sklearn.model_selection import KFold
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

        for train_idx, val_idx in kf.split(x_train):

            X_tr, X_val = x_train[train_idx], x_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            X_tr_red, _, mask = lasso_embedding(
                X_tr, y_tr, C=C, return_info=True)

            X_val_red = X_val[:, mask]

            model = LogisticRegressionModel()
            model.fit(X_tr_red, y_tr)

            scores.append(model.score(X_val_red, y_val))

        results[C] = np.mean(scores)      

        print(f"C = {C}, CV = {results[C]:.4f}")

    best_C = max(results, key=results.get)

    print(f"\nBest C: {best_C}")
    print(f"Best CV: {results[best_C]:.4f}")

    return best_C, results



# Evaluate model
def evaluate_model(model, x_test, y_test):

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# Main
def main():

    # Load original data
    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    # training_matrix = np.load("./data/train_matrix_0.5_flipped.npy")


    test_labels = np.load("./data/test_labels.npy")
    test_matrix = np.load("./data/test_matrix.npy")
    # test_matrix = np.load("./data/test_matrix_0.5_flipped.npy")

    # Number of features
    # feature_method = 'f_test'
    feature_method = 'lasso' # Behöver lasso = LogisticRegression(penalty="l1",C=C,solver="liblinear",max_iter=1000)i lasso_embedding
    flipped = False

    if feature_method == 'f_test':
        k_values = [100, 200, 500, 1000, 1500, 2000, 3000]

        # Find best k or C
        best_k, results = find_best_k_f_test(
            training_matrix,
            training_labels,
            k_values)

        # Feature selection on training and test data
        training_matrix_reduced, _, selected_pixel_idxs = f_score_filter(
            training_matrix, 
            training_labels, 
            k=best_k,
            return_scores=True)

        test_matrix_reduced = test_matrix[:, selected_pixel_idxs]

        print(f"Original number of features: {training_matrix.shape[1]}")
        print(f"Reduced number of features: {training_matrix_reduced.shape[1]}")

    elif feature_method == "lasso":
        C_values = [0.001, 0.01, 0.1, 1.0]

        best_C, results = find_best_C_lasso(
            training_matrix,
            training_labels,
            C_values)

        training_matrix_reduced, _, mask = lasso_embedding(
            training_matrix,
            training_labels,
            C=best_C,
            return_info=True)

        test_matrix_reduced = test_matrix[:, mask]

    else:
        raise ValueError("Feature method must be 'f_test' or 'lasso'")


    # Choose path name
    suffix = '_flipped' if flipped else ''
    path = f"./saved_models/logistic_regression_{feature_method}{suffix}.pkl"


    # Train model
    cv_score = train_logistic_regression(
        training_matrix_reduced,
        training_labels,
        n_folds=10,
        save_model=path)

    print(f"Cross-validation accuracy: {cv_score:.4f}")

    # Load saved model
    loaded_model = LogisticRegressionModel().load(path)
    
    # Evaluate on test data
    test_accuracy = evaluate_model(
        loaded_model,
        test_matrix_reduced,
        test_labels)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Model trained, saved, loaded, and evaluated successfully")


if __name__ == "__main__":
    main()