import numpy as np
import joblib
from train_utils import kCV, hyper_parameter_opt
from f_test_filter_selection import f_score_filter
import torch
from lasso_regression_selection import lasso_embedding

from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


# Wrapper class
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

    def train(self, train_data, val_data, k_val=None, lasso_c=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        x_train, y_train = train_data
        x_val, y_val = val_data

        if k_val:
            x_train, _, selected_idx = f_score_filter(
                x_train, y_train, k_val, return_scores=True
            )
            x_val = x_val[:, selected_idx]
        if lasso_c:
            x_train, _, selected_idx = lasso_embedding(
                x_train, y_train, C=lasso_c, return_info=True
            )
            x_val = x_val[:, selected_idx]
            print(x_train.shape)

        x_train, y_train = torch.from_numpy(x_train).to(device), torch.from_numpy(
            y_train
        ).to(device)

        self.fit(x_train, y_train)
        score = self.score(x_val, y_val)
        return score


# Train model
def train_XGB(x_train, y_train, settings, n_folds=10, save_model=None):

    xgb = XGBoost(**settings)

    # Calculate CV scores
    print("CV started...")
    scores = cross_val_score(
        xgb.model, x_train, y_train, cv=n_folds, scoring="accuracy", n_jobs=-1
    )
    print("CV finished.")

    # Training
    xgb.fit(x_train, y_train)

    if save_model:
        xgb.save(save_model)

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

    classifier_settings = {
        "n_estimators": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    training_settings = {"k_val": None, "lasso_c": 0.1}

    kCV_settings = {
        "k": 10,
        "model_class": XGBoost,
        "train_input": training_matrix,
        "train_target": training_labels,
        "model_settings": classifier_settings,
        "training_settings": training_settings,
        "save_model": False,
        "model_name": "XGBoost",
        "data_loaders": False,
    }
    # kCV(**kCV_settings)

    params = [classifier_settings, training_settings, kCV_settings]

    values = [1, 0.5, 0.25, 0.1, 0.05, 0.025, 0.001]
    hyper_parameter_opt("lasso_c", values, "train", params, "xgboost")

    # save_model="./saved_models/xgboost_flipped.pkl")

    # print(f"Cross-validation accuracy: {cv_score:.4f}")

    # loaded_model = XGBoost().load("./saved_models/xgboost.pkl")

    # "./saved_models/xgboost_flipped.pkl")

    # test_accuracy = evaluate_model(loaded_model, test_matrix, test_labels)

    # print(f"Test accuracy: {test_accuracy:.4f}")
    # print("XGBoost model trained, saved, loaded, and evaluated successfully")


if __name__ == "__main__":
    main()
