import numpy as np
from log_reg import LogRegAdapter, TorchTrainConfig, LogisticRegression
from train_pipeline import run_pipeline, ModelConfig, run_defect_pipeline, kCV_outer
from skopt.space import Integer, Real, Categorical
from data_process import get_data_balance
import torch


def main():
    hyper_params = [
        Real(0, 0.01, name="l2"),
    ]
    model_params = {
        "in_features": 19,
        "number_of_classes": 7,
    }
    # sizes = [1, 3]

    # for size in sizes:

    sample = 7680

    train_inputs = np.load(f"./data/{sample}_input.npy")
    train_labels = np.load(f"./data/{sample}_labels.npy")

    imbalance = torch.tensor(get_data_balance(train_labels)).float()

    train_config = TorchTrainConfig(hyper_params, class_imbalance=imbalance)
    model_config = ModelConfig(
        "Logistic Regression",
        LogisticRegression,
        model_params,
        train_config,
    )
    model_adapter = LogRegAdapter(model_config, f"./models/1d/LogReg_{sample}")

    train_data = [train_inputs, train_labels]
    kCV_outer(model_adapter, train_data, multiple_runs=0)

    # run_pipeline(model_adapter, train_data)
    # run_defect_pipeline(model_adapter, train_data, size)


if __name__ == "__main__":
    main()
