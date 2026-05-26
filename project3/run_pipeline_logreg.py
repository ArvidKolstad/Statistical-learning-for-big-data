import pandas as pd
from log_reg import LogRegAdapter, LogisticRegression, TorchTrainConfig
from train_pipeline import ModelConfig, run_pipeline
from skopt.space import Integer


def main():
    hyper_params_search = [
        Integer(10, 100, name="in_features"),
    ]
    train_config = TorchTrainConfig(hyper_params_search)
    model_settings = {
        "in_features": 500,
    }
    model_config = ModelConfig(
        "Logistic Regressio",
        LogisticRegression,
        model_settings,
        train_config,
    )
    model_adapter = LogRegAdapter(model_config, "./models/LogReg_250_imbalance")

    df = pd.read_csv("./data/data_cat50_dog200.csv")

    out_data = df["label"].to_numpy()

    in_data = df.drop(columns="label").to_numpy(dtype=float)
    data = [in_data, out_data]
    run_pipeline(model_adapter, data)


if __name__ == "__main__":
    main()
