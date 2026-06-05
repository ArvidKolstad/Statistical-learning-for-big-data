import numpy as np
import torch
import pandas as pd

from reg_disc import RDAModelAdapter, RDATrainConfig, RegularizedDiscriminantAnalysis
from knn import KNNModelAdapter, KNNClassifier
from log_reg import LogRegAdapter, TorchTrainConfig, LogisticRegression

from train_pipeline import get_mislabeling, ModelConfig, BaseTrainConfig
from skopt.space import Integer, Real, Categorical
from data_process import get_data_balance


def mislabel_data():
    inputs = np.load("./data/200_input.npy")
    labels = np.load("./data/200_labels.npy")
    unique_labels = np.unique(labels)

    new_labels = np.roll(unique_labels, shift=1)

    label_map = dict(zip(unique_labels, new_labels))

    vectorized_map = np.vectorize(lambda x: label_map[x])
    labels = vectorized_map(labels)
    return inputs, labels


def to_tuple_key(arr):
    return tuple(arr.astype(np.float32).flatten())


def main():
    sample = 7680

    # data
    train_inputs = np.load(f"./data/{sample}_input.npy")
    train_labels = np.load(f"./data/{sample}_labels.npy")
    imbalance = torch.tensor(get_data_balance(train_labels)).float()
    train_data = [train_inputs, train_labels]
    wrong_data = mislabel_data()

    # logreg config
    hyper_params_logreg = [
        Real(0, 0.01, name="l2"),
    ]
    model_params_logreg = {
        "in_features": 20,
        "number_of_classes": 7,
    }

    train_config_logreg = TorchTrainConfig(
        hyper_params_logreg, class_imbalance=imbalance
    )
    model_config_logreg = ModelConfig(
        "Logistic Regression",
        LogisticRegression,
        model_params_logreg,
        train_config_logreg,
    )
    model_adapter_logreg = LogRegAdapter(
        model_config_logreg, f"./models/1c/LogReg_{sample}", check_mislabeling=True
    )

    # knn
    hyper_params_knn = [
        Integer(10, 50, name="n_neighbors"),
        Real(1.0, 2.0, name="p"),
        Categorical(["uniform", "distance"], name="weights"),
    ]
    train_config_knn = BaseTrainConfig(hyper_params_knn)

    model_settings_knn = {
        "in_features": 20,
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "n_jobs": -1,
    }

    model_config_knn = ModelConfig(
        "KNN",
        KNNClassifier,
        model_settings_knn,
        train_config_knn,
    )

    model_adapter_knn = KNNModelAdapter(
        model_config_knn, f"./models/1c/KNN_{sample}", check_mislabeling=True
    )

    # RDA
    hyper_params_rda = [
        Real(0, 1, name="lmbda"),
        Real(0, 1, name="gamma"),
    ]
    model_params_rda = {
        "in_features": 20,
        "classes": [0, 1, 2, 3, 4, 5, 6],
        "lmbda": 0.5,
        "gamma": 0.5,
    }
    train_config_rda = RDATrainConfig(hyper_params_rda)
    model_config_rda = ModelConfig(
        "Regularized Discriminant Analysis",
        RegularizedDiscriminantAnalysis,
        model_params_rda,
        train_config_rda,
    )
    model_adapter_rda = RDAModelAdapter(
        model_config_rda, f"./models/1c/RDA_{sample}", check_mislabeling=True
    )

    mislabeled_rda, wrong_data_rda = get_mislabeling(
        model_adapter_rda, train_data, wrong_data
    )
    mislabeled_logreg, wrong_data_logreg = get_mislabeling(
        model_adapter_logreg, train_data, wrong_data
    )
    mislabeled_knn, wrong_data_knn = get_mislabeling(
        model_adapter_knn, train_data, wrong_data
    )

    df_rda = pd.DataFrame(mislabeled_rda)
    df_logreg = pd.DataFrame(mislabeled_logreg)
    df_knn = pd.DataFrame(mislabeled_knn)

    df_rda["match_key"] = df_rda["input value"].apply(to_tuple_key)
    df_logreg["match_key"] = df_logreg["input value"].apply(to_tuple_key)
    df_knn["match_key"] = df_knn["input value"].apply(to_tuple_key)

    is_in_logreg = df_rda["match_key"].isin(df_logreg["match_key"])
    is_in_knn = df_rda["match_key"].isin(df_knn["match_key"])

    common_mislabeled_df = df_rda[is_in_logreg & is_in_knn]
    common_mislabeled_df = common_mislabeled_df.drop(columns=["match_key"])
    print(f"Mislabels: {common_mislabeled_df.shape[0]}")
    common_mislabeled_df.to_csv("./figures/problem3/mislabeled.csv")

    df_rda = pd.DataFrame(wrong_data_rda)
    df_logreg = pd.DataFrame(wrong_data_logreg)
    df_knn = pd.DataFrame(wrong_data_knn)

    df_rda["match_key"] = df_rda["input value"].apply(to_tuple_key)
    df_logreg["match_key"] = df_logreg["input value"].apply(to_tuple_key)
    df_knn["match_key"] = df_knn["input value"].apply(to_tuple_key)

    is_in_logreg = df_rda["match_key"].isin(df_logreg["match_key"])
    is_in_knn = df_rda["match_key"].isin(df_knn["match_key"])

    common_mislabeled_df = df_rda[is_in_logreg & is_in_knn]
    common_mislabeled_df = common_mislabeled_df.drop(columns=["match_key"])

    print(f"Found wrongly labeld : {common_mislabeled_df.shape[0]}/200")


if __name__ == "__main__":
    main()
