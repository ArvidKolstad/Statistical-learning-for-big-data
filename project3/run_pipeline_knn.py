from skopt.space import Integer, Real, Categorical
from knn import KNNModelAdapter, KNNClassifier
import pandas as pd
from train_pipeline import BaseTrainConfig, run_pipeline, ModelConfig


def main():

    hyper_params_search = [
        Integer(10, 100, name="in_features"),
        Integer(10, 500, name="n_neighbors"),
        Real(1.0, 2.0, name="p"),
        Categorical(["uniform", "distance"], name="weights"),
    ]
    train_config = BaseTrainConfig(hyper_params_search)

    model_settings = {
        "in_features": 50,
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
    model_adapter = KNNModelAdapter(model_config, "./models/KNN_1000")

    df = pd.read_csv("./data/data_1000.csv")

    out_data = df["label"].to_numpy()
    in_data = df.drop(columns="label").to_numpy()

    data = [in_data, out_data]
    run_pipeline(model_adapter, data)


if __name__ == "__main__":
    main()
