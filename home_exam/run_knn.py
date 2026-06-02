import numpy as np
from knn import KNNModelAdapter, KNNClassifier
from train_pipeline import (
    run_pipeline,
    ModelConfig,
    BaseTrainConfig,
    run_defect_pipeline,
)
from skopt.space import Integer, Real, Categorical


def main():

    hyper_params = [
        Integer(10, 50, name="n_neighbors"),
        Real(1.0, 2.0, name="p"),
        Categorical(["uniform", "distance"], name="weights"),
    ]
    train_config = BaseTrainConfig(hyper_params)

    model_settings = {
        "in_features": 19,
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "n_jobs": -1,
    }

    model_config = ModelConfig(
        "KNN",
        KNNClassifier,
        model_settings,
        train_config,
    )
    sizes = [5, 15, 20, 40, 60]

    for size in sizes:
        sample = 7680

        model_adapter = KNNModelAdapter(model_config, f"./models/1b/KNN_{size}")
        train_inputs = np.load(f"./data/{sample}_input.npy")
        train_labels = np.load(f"./data/{sample}_labels.npy")

        train_data = [train_inputs, train_labels]

        # run_pipeline(model_adapter, train_data)
        run_defect_pipeline(model_adapter, train_data, size)


if __name__ == "__main__":
    main()
