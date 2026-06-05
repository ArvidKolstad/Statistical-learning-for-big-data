import numpy as np
import pickle as pkl
from predict_pipeline import get_model_probability_performance, predict_testset
from knn import KNNModelAdapter, KNNClassifier
from train_pipeline import (
    run_pipeline,
    ModelConfig,
    BaseTrainConfig,
    run_defect_pipeline,
    kCV_outer,
)
from skopt.space import Integer, Real, Categorical


def main():

    hyper_params = [
        Integer(10, 50, name="n_neighbors"),
        Real(1.0, 2.0, name="p"),
        Categorical(["uniform", "distance"], name="weights"),
    ]
    train_config = BaseTrainConfig(hyper_params)

    model_params = {
        "in_features": 20,
        "n_neighbors": np.int64(10),
        "weights": np.str_("distance"),
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 1.0237261075936646,
        "metric": "minkowski",
        "n_jobs": -1,
    }

    model_config = ModelConfig(
        "KNN",
        KNNClassifier,
        model_params,
        train_config,
    )
    sizes = [1, 3, 5, 15, 20, 40]

    # for size in sizes:
    sample = 7680

    samples = [200, 500, 1000, 3000, 5000, 7680]

    splits = 10
    # for sample in samples:
    model_adapter = KNNModelAdapter(
        model_config, f"./models/1d/KNN_{sample}", save_configs=True
    )
    train_inputs = np.load(f"./data/{sample}_input.npy")
    train_labels = np.load(f"./data/{sample}_labels.npy")
    test_data = np.load(f"./data/test_set.npy")

    train_data = [train_inputs, train_labels]
    #run_pipeline(model_adapter, train_data)
    # run_defect_pipeline(model_adapter, train_data, size)

    kCV_outer(model_adapter, train_data, multiple_runs=0)
    with open(model_adapter.output_dir + "config.pickle", "rb") as f:
        model_adapter.config.hyperparameters = pkl.load(f)

    get_model_probability_performance(model_adapter, train_data, splits)
    # predict_testset(model_adapter, train_data, test_data)


if __name__ == "__main__":
    main()
