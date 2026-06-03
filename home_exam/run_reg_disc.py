import numpy as np
import pickle as pkl
from predict_pipeline import get_model_probability_performance, predict_testset
from reg_disc import RDAModelAdapter, RDATrainConfig, RegularizedDiscriminantAnalysis
from train_pipeline import run_pipeline, ModelConfig, run_defect_pipeline, kCV_outer
from skopt.space import Integer, Real, Categorical


def main():
    hyper_params = [
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
    ]

    model_params = {
        "in_features": 19,
        "classes": [0, 1, 2, 3, 4, 5, 6],
        "lmbda": 0.19788427205107756,
        "gamma": 0.0026512062937177343,
    }
    train_config = RDATrainConfig(hyper_params)
    model_config = ModelConfig(
        "Regularized Discriminant Analysis",
        RegularizedDiscriminantAnalysis,
        model_params,
        train_config,
    )
    # sizes = [1, 3]

    samples = [200, 500, 1000, 3000, 5000, 7680]

    splits = 10

    for sample in samples:

        model_adapter = RDAModelAdapter(
            model_config, f"./models/1d/RDA_{sample}", save_configs=True
        )
        train_inputs = np.load(f"./data/{sample}_input.npy")
        train_labels = np.load(f"./data/{sample}_labels.npy")
        test_data = np.load(f"./data/test_set.npy")
        train_data = [train_inputs, train_labels]

        # run_pipeline(model_adapter, train_data)
        # run_defect_pipeline(model_adapter, train_data, size)

        # kCV_outer(model_adapter, train_data, multiple_runs=0)
        with open(model_adapter.output_dir + "config.pickle", "rb") as f:
            model_adapter.config.hyperparameters = pkl.load(f)

        # get_model_probability_performance(model_adapter, train_data, splits)
        predict_testset(model_adapter, train_data, test_data)


if __name__ == "__main__":
    main()
