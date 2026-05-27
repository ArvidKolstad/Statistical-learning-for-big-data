import pandas as pd
from train_pipeline import ModelConfig, run_pipeline
from xgb import XGBTrainConfig, XGBModelAdapter, XGBoostClassifier
from skopt.space import Integer, Real


def main():

    hyper_params_search = [
        Integer(10, 100, name="in_features"),
        Integer(10, 500, name="n_estimators"),
        Real(0, 0.1, name="learning_rate"),
        Real(0.4, 1.0, name="subsample"),
        Real(0.4, 1.0, name="colsample_bytree"),
    ]

    train_config = XGBTrainConfig(hyper_params_search)

    model_settings = {
        "in_features": 50,
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "random_state": 42,
        "device": "cuda",
    }

    model_config = ModelConfig(
        "XGBoost Classifer",
        XGBoostClassifier,
        model_settings,
        train_config,
    )
    model_adapter = XGBModelAdapter(model_config, "./models/XGB_1000_random")

    df = pd.read_csv("./data/data_random.csv")

    out_data = df["label"].to_numpy()
    in_data = df.drop(columns="label").to_numpy()

    data = [in_data, out_data]
    run_pipeline(model_adapter, data)


if __name__ == "__main__":
    main()
