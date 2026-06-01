import numpy as np
from xgb import XGBTrainConfig, XGBModelAdapter, XGBoostClassifier
from train_pipeline import run_pipeline, ModelConfig
from skopt.space import Integer, Real, Categorical
from data_process import get_data_balance
import torch


def main():
    hyper_params = [
        Integer(10, 820, name="in_features"),
        Integer(10, 500, name="n_estimators"),
        Real(0, 0.01, name="learning_rate"),
        Real(0.0, 1.0, name="subsample"),
        Real(0.0, 1.0, name="colsample_bytree"),
    ]

    model_settings = {
        "in_features": 50,
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "multi:softmax",
        "random_state": 42,
        "device": "cuda",
    }

    train_inputs = np.load("./data/7680_input.npy")
    train_labels = np.load("./data/7680_labels.npy")

    train_config = XGBTrainConfig(hyper_params)
    model_config = ModelConfig(
        "Logistic Regression",
        XGBoostClassifier,
        model_settings,
        train_config,
    )
    model_adapter = XGBModelAdapter(model_config, "./models/LogReg_7680")

    train_data = [train_inputs, train_labels]

    run_pipeline(model_adapter, train_data)


if __name__ == "__main__":
    main()
