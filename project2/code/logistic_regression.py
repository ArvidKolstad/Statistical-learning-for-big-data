import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Train model
def train_logistic_regression(x_train, y_train, n_folds=10):

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ))
    ])

    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        cv=n_folds,
        scoring='accuracy',
        n_jobs=-1
    )

    pipeline.fit(x_train, y_train)

    return pipeline, scores.mean()


# Save model
def save_model(model, filename):
    joblib.dump(model, filename)


# Load model
def load_model(filename):
    return joblib.load(filename)


# Evaluate model
def evaluate_model(model, x_test, y_test):

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    return accuracy


def main():

    training_labels = np.load("./data/train_labels.npy")
    # training_matrix = np.load("./data/train_matrix.npy")
    training_matrix = np.load("./data/train_matrix_0.5_flipped.npy")


    test_labels = np.load("./data/test_labels.npy")
    # test_matrix = np.load("./data/test_matrix.npy")
    test_matrix = np.load("./data/test_matrix_0.5_flipped.npy")

    model, cv_score = train_logistic_regression(
        training_matrix,
        training_labels,
        n_folds=10
    )

    print(f"Cross-validation accuracy: {cv_score:.4f}")

    # save_model(model, "./saved_models/logistic_regression.pkl")
    save_model(model, "./saved_models/logistic_regression_flipped.pkl")

    # loaded_model = load_model("./saved_models/logistic_regression.pkl")
    loaded_model = load_model("./saved_models/logistic_regression_flipped.pkl")

    test_accuracy = evaluate_model(
        loaded_model,
        test_matrix,
        test_labels
    )

    print(f"Test accuracy: {test_accuracy:.4f}")

    print("Model trained, saved, loaded, and evaluated successfully")


if __name__ == "__main__":
    main()