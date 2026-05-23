import numpy as np
from log_reg import LogRegAdapter, LogisticRegressionModel
from train_pipeline import ModelConfig, run_pipeline, BaseTrainConfig
from skopt.space import Integer


def main():

    hyper_params_search = [
        Integer(10, 1000, name="in_features"),
    ]
    train_config = BaseTrainConfig(hyper_params_search)
    model_settings = {
        "in_features": 500,
    }
    model_config = ModelConfig(
        "Logistic Regressio",
        LogisticRegressionModel,
        model_settings,
        train_config,
    )
    model_adapter = LogRegAdapter(model_config, "./models/LogReg")

    in_data = np.load("./data/train_matrix.npy")
    out_data = np.load("./data/train_labels.npy")
    data = [in_data, out_data]
    run_pipeline(model_adapter, data)


if __name__ == "__main__":
    main()
