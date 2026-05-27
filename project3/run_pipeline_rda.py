import numpy as np
import pandas as pd
from reg_disc import RDAModelAdapter, RDATrainConfig, RegularizedDiscriminantAnalysis
from train_pipeline import ModelConfig, run_pipeline
from skopt.space import Integer, Real, Categorical


def main():

    hyper_params_search = [
        Integer(10, 100, name="in_features"),
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
    ]
    train_config = RDATrainConfig(hyper_params_search)
    model_settings = {
        "in_features": 500,
        "classes": 2,
        "lmbda": 0.5,
        "gamma": 0.5,
    }
    model_config = ModelConfig(
        "Regularized Discriminant Analysis",
        RegularizedDiscriminantAnalysis,
        model_settings,
        train_config,
    )
    model_adapter = RDAModelAdapter(model_config, "./models/RDA_500")

    df = pd.read_csv("./data/data_500.csv")
    out_data = df["label"].to_numpy()
    in_data = df.drop(columns="label").to_numpy()

    data = [in_data, out_data]
    run_pipeline(model_adapter, data)


if __name__ == "__main__":
    main()
