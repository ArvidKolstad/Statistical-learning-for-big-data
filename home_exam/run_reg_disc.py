import numpy as np
from reg_disc import RDAModelAdapter, RDATrainConfig, RegularizedDiscriminantAnalysis
from train_pipeline import run_pipeline, ModelConfig, run_defect_pipeline
from skopt.space import Integer, Real, Categorical


def main():
    hyper_params = [
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
    ]
    model_params = {
        "in_features": 19,
        "classes": [0, 1, 2, 3, 4, 5, 6],
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
    sizes = [5, 15, 20, 40, 60]

    for size in sizes:
        sample = 7680

        model_adapter = RDAModelAdapter(model_config, f"./models/1b/RDA_{size}")
        train_inputs = np.load(f"./data/{sample}_input.npy")
        train_labels = np.load(f"./data/{sample}_labels.npy")
        train_data = [train_inputs, train_labels]

        # run_pipeline(model_adapter, train_data)
        run_defect_pipeline(model_adapter, train_data, size)


if __name__ == "__main__":
    main()
