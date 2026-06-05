from hierarchy_model import HierarchyModelAdapter, HierarchyModel
import numpy as np
import pickle as pkl
from train_pipeline import (
    run_pipeline,
    ModelConfig,
    run_defect_pipeline,
    kCV_outer,
    BaseTrainConfig,
)
from skopt.space import Integer, Real, Categorical


def main():
    hyper_params = [
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
        Integer(10, 50, name="n_neighbors"),
        Real(1.0, 2.0, name="p"),
        Categorical(["uniform", "distance"], name="weights"),
    ]

    model_params = {
        "in_features": 19,
        "classes_minority": [0, 1, 2, 3, 4, 5],
        "lmbda": 0.19788427205107756,
        "gamma": 0.0026512062937177343,
        "weights": "macro",
        "n_neighbors": np.int64(10),
        "weights": np.str_("distance"),
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 1.0237261075936646,
        "metric": "minkowski",
        "n_jobs": -1,
    }
    train_config = BaseTrainConfig(hyper_params)

    model_config = ModelConfig(
        "Hierarchy Model",
        HierarchyModel,
        model_params,
        train_config,
    )
    # sizes = [1, 3, 5, 15, 20, 40, 60]

    # samples = [200, 500, 1000, 3000, 5000, 7680]
    sample = 7680

    # splits = 10

    # for size in sizes:

    model_adapter = HierarchyModelAdapter(model_config, f"./models/2a/HierarchyModel")
    train_inputs = np.load(f"./data/extreme_inputs.npy")
    train_labels = np.load(f"./data/extreme_labels.npy")

    train_data = [train_inputs, train_labels]

    run_pipeline(model_adapter, train_data)
    # run_defect_pipeline(model_adapter, train_data, size)


if __name__ == "__main__":
    main()
