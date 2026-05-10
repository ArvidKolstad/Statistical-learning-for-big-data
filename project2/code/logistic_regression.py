import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Wrapper class
class LogisticRegressionModel:
    def __init__(self, **settings):
        self.model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=42,  # Bör kanske ändras sen för repetering
                    ),
                ),
            ]
        )

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
def train_logistic_regression(x_train, y_train, n_folds=10, save_model=None):

    model = LogisticRegressionModel()

    print("CV started...")
    scores = cross_val_score(
        model.model, x_train, y_train, cv=n_folds, scoring="accuracy", n_jobs=-1
    )
    print("CV finished.")

    model.fit(x_train, y_train)

    if save_model:
        model.save(save_model)

    return scores.mean()


# Evaluate model
def evaluate_model(model, x_test, y_test):

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy


# Main
def main():

    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    # training_matrix = np.load("./data/train_matrix_0.5_flipped.npy")

    test_labels = np.load("./data/test_labels.npy")
    test_matrix = np.load("./data/test_matrix.npy")
    # test_matrix = np.load("./data/test_matrix_0.5_flipped.npy")

    cv_score = train_logistic_regression(
        training_matrix,
        training_labels,
        n_folds=10,
        save_model="./saved_models/logistic_regression.pkl",
    )
    # save_model="./saved_models/logistic_regression_flipped.pkl")

    print(f"Cross-validation accuracy: {cv_score:.4f}")

    loaded_model = LogisticRegressionModel().load(
        "./saved_models/logistic_regression.pkl"
    )
    # loaded_model = LogisticRegression().load(
    #     "./saved_models/logistic_regression_flipped.pkl")

    test_accuracy = evaluate_model(loaded_model, test_matrix, test_labels)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Model trained, saved, loaded, and evaluated successfully")


if __name__ == "__main__":
    main()

