import numpy as np
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, SubsetRandomSampler
import matplotlib.pyplot as plt


def kCV(
    k,
    model_class,
    data_set,
    model_settings,
    training_settings,
    save_model=False,
    model_name="normal",
):

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    best_fold_score = 0.0
    fold_score = []

    matrixes = data_set
    labels = data_set

    for fold, (train_idx, val_idx) in enumerate(skf.split(matrixes, labels)):
        print(f"Fold: {fold +1 }/{k}")

        train_data_loader = DataLoader(
            data_set, batch_size=100, sampler=SubsetRandomSampler(train_idx)
        )

        val_data_loader = DataLoader(
            data_set, batch_size=256, sampler=SubsetRandomSampler(val_idx)
        )

        training_settings["train_data"] = train_data_loader
        training_settings["val_data"] = val_data_loader

        model = model_class(**model_settings)

        score = model.train(**training_settings)

        if score > best_fold_score and save_model:
            best_fold_score = score

            model.save("./saved_models/" + model_name)
            model.save_model_settings(model_name)

        fold_score.append(score)
        print(f"Best val score: {score}")
    return fold_score


def get_dict(module):
    if module == "data":
        return 0
    elif module == "train":
        return 1
    else:
        raise ValueError("Module doesn't exist")


def hyper_parameter_opt(
    hyper_parameter: str,
    parameter_values: list,
    module: str,
    params: list,
    type_run,
    data_matrix=None,
    data_label=None,
):

    print("Now training for " + hyper_parameter)
    if isinstance(params[get_dict(module)][hyper_parameter], (list, np.ndarray)):
        original_value = params[get_dict(module)][hyper_parameter].copy()
    else:
        original_value = params[get_dict(module)][hyper_parameter]

    original_data_set = params[get_dict("data")]
    color = ["red", "blue"]
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    if data_matrix is None and data_label is None:
        redo_data = False
    else:
        redo_data = True

    input_dim = None
    x1 = None
    value = None
    scores = []
    x2 = []

    for idx, value in enumerate(parameter_values):

        params[get_dict(module)][hyper_parameter] = value
        score = kCV(*params)
        scores.append(score)
        if hyper_parameter == "layer_dim":
            x1 = np.ones_like(score) * input_dim
            x2.append(input_dim)
        else:
            x1 = np.ones_like(score) * value
            x2.append(value)

        ax1.scatter(x1, score, color=color[idx % 2])

        print(f"Done hyper training: {idx+1}/{len(parameter_values)}")

    ax2.boxplot(scores, tick_labels=x2)
    ax2.set_title(f"Hyper parameter: {hyper_parameter}")
    fig2.savefig(
        "../figures/hyper_param_tune_mlp/" + hyper_parameter + "_boxplot" + type_run
    )

    ax1.grid()
    ax1.set_title(f"Hyper parameter: {hyper_parameter}")
    fig1.tight_layout()
    fig1.savefig("../figures/hyper_param_tune_mlp/" + hyper_parameter + type_run)
    params[get_dict(module)][hyper_parameter] = original_value

    if redo_data:
        params[get_dict("data")] = original_data_set
