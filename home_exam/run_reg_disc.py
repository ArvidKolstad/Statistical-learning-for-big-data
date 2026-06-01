import numpy as np
from reg_disc import RDAModelAdapter, RDATrainConfig, RegularizedDiscriminantAnalysis
from train_pipeline import run_pipeline, ModelConfig
from skopt.space import Integer, Real, Categorical


def main():
    hyper_params = [
        Integer(10, 820, name="in_features"),
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
    ]
    model_params = {
        "in_features": 100,
        "classes": [1, 2, 3, 4, 5, 6, 7],
        "lmbda": 0.5,
        "gamma": 0.5,
    }
    train_config = RDATrainConfig(hyper_params)
    model_config = ModelConfig(
        "Regularized Discriminant Analysis",
        RegularizedDiscriminantAnalysis,
        model_params,
        train_config,
    )
    model_adapter = RDAModelAdapter(model_config, "./models/RDA_7680")
    train_inputs = np.load("./data/7680_input.npy")
    train_labels = np.load("./data/7680_labels.npy")
    train_data = [train_inputs, train_labels]

    run_pipeline(model_adapter, train_data)


if __name__ == "__main__":
    main()
