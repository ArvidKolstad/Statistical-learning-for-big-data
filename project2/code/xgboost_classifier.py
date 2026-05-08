import numpy as np
import joblib

from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Wrapper object
class XGBoost:
    def __init__(self, **settings):
        self.model = XGBClassifier(**settings)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def score(self, X, y):
        return self.model.score(X, y)

    def save(self, filename):
        joblib.dump(self.model, filename)

    def load(self, filename):
        self.model = joblib.load(filename)
        return self


# Train model
def train_XGB(
    x_train,
    y_train,
    settings,
    n_folds=10,
    save_model=None):

    xgb = XGBoost(**settings)

    print('CV started...')
    scores = cross_val_score(
        xgb.model,
        x_train,
        y_train,
        cv=n_folds,
        scoring='accuracy',
        n_jobs=-1)
    print('CV finished.')

    xgb.fit(x_train, y_train)

    if save_model:
        xgb.save(save_model)

    return xgb, scores.mean()


# Evaluate model
def evaluate_model(
    model,
    x_test, 
    y_test):

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return accuracy



def main():

    training_labels = np.load("./data/train_labels.npy")
    training_matrix = np.load("./data/train_matrix.npy")
    # training_matrix = np.load("./data/train_matrix_0.5_flipped.npy")


    test_labels = np.load("./data/test_labels.npy")
    test_matrix = np.load("./data/test_matrix.npy")
    # test_matrix = np.load("./data/test_matrix_0.5_flipped.npy")


    classifier_settings = {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "random_state": 42, # Ska kanske ändras sen när det ska repeteras
        "n_jobs": -1}

    model, cv_score = train_XGB(
        training_matrix,
        training_labels,
        classifier_settings,
        n_folds=10,
        save_model="./saved_models/xgboost.pkl")
        # save_model="./saved_models/xgboost_flipped.pkl")

    print(f'Cross-validation accuracy: {cv_score:.4f}')

    loaded_model = XGBoost().load(
        "./saved_models/xgboost.pkl")
        # "./saved_models/xgboost_flipped.pkl")

    test_accuracy = evaluate_model(
        loaded_model,
        test_matrix,
        test_labels)


    print(f"Test accuracy: {test_accuracy:.4f}")
    print("XGBoost model trained, saved, loaded, and evaluated successfully")


if __name__ == "__main__":
    main()