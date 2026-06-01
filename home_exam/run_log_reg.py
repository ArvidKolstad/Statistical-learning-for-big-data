import numpy as np
from log_reg import LogRegAdapter, TorchTrainConfig, LogisticRegression
from train_pipeline import run_pipeline, ModelConfig
from skopt.space import Integer, Real, Categorical
from data_process import get_data_balance
import torch


def main():
    hyper_params = [
        Integer(10, 820, name="in_features"),
        Real(0, 0.01, name="l2"),
    ]
    model_params = {
        "in_features": 100,
        "number_of_classes": 7,
    }
    train_inputs = np.load("./data/7680_input.npy")
    train_labels = np.load("./data/7680_labels.npy")

    imbalance = torch.tensor(get_data_balance(train_labels)).float()

    train_config = TorchTrainConfig(hyper_params, class_imbalance=imbalance)
    model_config = ModelConfig(
        "Logistic Regression",
        LogisticRegression,
        model_params,
        train_config,
    )
    model_adapter = LogRegAdapter(model_config, "./models/LogReg_7680")

    train_data = [train_inputs, train_labels]

    run_pipeline(model_adapter, train_data)


if __name__ == "__main__":
    main()
