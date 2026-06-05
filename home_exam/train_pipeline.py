import numpy as np
import pickle as pkl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from dataclasses import dataclass, field
from typing import Any, Type, Optional, Generic, TypeVar
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical


@dataclass
class BaseTrainConfig:
    search_space: list[Any]
    seed: list[int] = field(
        default_factory=lambda: [6, 7, 42, 10, 57, 67, 69, 103, 43, 37]
    )
    R: int = 10
    K: int = 10
    L: int = 10


T_Config = TypeVar("T_Config", bound=BaseTrainConfig)


@dataclass
class ModelConfig(Generic[T_Config]):
    name: "str"
    model_class: Type[Any]
    hyperparameters: dict[str, Any]
    training_settings: T_Config


class BaseModelAdapter(Generic[T_Config]):
    def __init__(
        self,
        config: ModelConfig[T_Config],
        save_path: str,
        check_mislabeling=False,
        save_configs=False,
    ):
        self.config = config
        self.model = config.model_class(**config.hyperparameters)
        self.output_dir: str = save_path
        self.check_mislabeling = check_mislabeling
        self.save_configs: bool = save_configs

    def clean_model(self):
        self.model = self.config.model_class(**self.config.hyperparameters)

    def train_params(self, train_batch) -> None:
        raise NotImplementedError

    def validate(self, val_batch) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def get_class(self, val_input) -> np.ndarray:
        raise NotImplementedError

    def get_probability(self, val_input) -> np.ndarray:
        raise NotImplementedError

    def save_model(self) -> None:
        raise NotImplementedError

    def load_model(self) -> None:
        raise NotImplementedError


def kCV_inner(
    model_adapter: BaseModelAdapter, hyper_opt_data, outer_fold: Optional[int] = None
) -> np.float32:
    training_settings = model_adapter.config.training_settings

    skf = StratifiedKFold(n_splits=training_settings.L, shuffle=True)

    if outer_fold:
        skf.random_state = training_settings.seed[outer_fold % training_settings.R]

    fold_score = np.zeros(training_settings.L)

    for fold, (train_idx, val_idx) in enumerate(skf.split(*hyper_opt_data)):
        train_batch = [values[train_idx] for values in hyper_opt_data]
        val_batch = [values[val_idx] for values in hyper_opt_data]

        model_adapter.clean_model()

        model_adapter.train_params(train_batch)
        preds, labels = model_adapter.validate(val_batch)

        correct_classification = np.sum(labels == preds)
        total_classification = labels.shape[0]
        accuracy = correct_classification / total_classification

        fold_score[fold] = accuracy
    mean_score = np.mean(fold_score)

    return mean_score


def hyper_parameter_opt(
    model_adapter: BaseModelAdapter, hyper_opt_data, outer_fold: Optional[int]
):
    training_settings = model_adapter.config.training_settings
    opt = Optimizer(
        dimensions=training_settings.search_space, base_estimator="GP", random_state=42
    )
    for i in range(20):
        print(f"Now running hyperparameter opt {i+1}/20")

        next_config = opt.ask()

        if isinstance(next_config, list):
            for dimension, val in zip(training_settings.search_space, next_config):
                model_adapter.config.hyperparameters[dimension.name] = val

        else:
            raise TypeError(
                f"Unexpected return type from opt.ask(): {type(next_config)}"
            )
        mean_score = kCV_inner(model_adapter, hyper_opt_data, outer_fold)
        cost = -mean_score
        opt.tell(next_config, cost)
    results = opt.get_result()

    for hyper_param, value in zip(opt.space.dimension_names, results.x):
        model_adapter.config.hyperparameters[hyper_param] = value


def check_mislabel(model_adapter, val_batch, threshold=0.05) -> list[dict]:
    val_input, val_labels = val_batch
    probs = model_adapter.get_probability(val_input)
    outputs, val_labels = model_adapter.validate(val_batch)
    suspicious_samples = []

    for idx, (sample, label) in enumerate(zip(probs, val_labels)):
        assert np.argmax(sample) == outputs[idx]
        if sample[int(label)] < threshold:
            suspicious_samples.append(
                {
                    "input value": val_input[idx],
                    "guessed label": outputs[idx],
                    "actual label": label,
                }
            )
    return suspicious_samples


def kCV_outer(
    model_adapter: BaseModelAdapter,
    data: list[np.ndarray],
    multiple_runs: Optional[int],
    wrong_data=[],
):
    best_score = 0.0

    training_settings = model_adapter.config.training_settings

    skf = StratifiedKFold(n_splits=training_settings.K, shuffle=True)

    if multiple_runs:
        skf.random_state = training_settings.seed[multiple_runs]

    fold_scores = np.zeros((training_settings.K, 2))
    x, y = data
    fold_acc = np.zeros((training_settings.K, 7))
    mislabeled_data = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(x, y)):
        class_based_accuracy = []

        print(f"Outer fold: {fold +1 }/{training_settings.K}")
        train_batch = [values[train_idx] for values in data]
        val_batch = [values[val_idx] for values in data]

        hyper_parameter_opt(model_adapter, train_batch, fold)
        model_adapter.clean_model()

        model_adapter.train_params(train_batch)

        preds, labels = model_adapter.validate(val_batch)

        for cls in np.unique(labels):
            mask = cls == labels
            class_based_accuracy.append(np.sum(preds[mask] == cls) / np.sum(mask))

        correct_classification = np.sum(labels == preds)
        total_classification = labels.shape[0]

        accuracy = correct_classification / total_classification
        f1 = f1_score(labels, preds, average="macro")

        scores = np.array([accuracy, f1])

        fold_scores[fold] = scores
        fold_acc[fold] = class_based_accuracy
        if model_adapter.check_mislabeling:
            suspicious_samples = check_mislabel(model_adapter, val_batch)

            mislabeled_data += suspicious_samples
        if model_adapter.save_configs and (f1 > best_score):
            with open(model_adapter.output_dir + "config.pickle", "wb") as f:
                pkl.dump(
                    model_adapter.config.hyperparameters,
                    f,
                    protocol=pkl.HIGHEST_PROTOCOL,
                )

    if model_adapter.check_mislabeling:
        print(len(wrong_data))
        wrong_samples = check_mislabel(model_adapter, wrong_data)

        return mislabeled_data, wrong_samples

    return fold_scores, fold_acc


def get_mislabeling(
    model_adapter: BaseModelAdapter, data: list[np.ndarray], wrong_data
):
    training_settings = model_adapter.config.training_settings
    mislabels = []
    wrong_samples = []
    for R in range(training_settings.R):
        print(f"Now running R: {R+1}/{training_settings.R}")
        mislabel_data, wrong_samples = kCV_outer(
            model_adapter, data, R, wrong_data=wrong_data
        )
        for new_sample in mislabel_data:
            already_sus = False
            for sample in mislabels:
                if np.array_equal(sample["input value"], new_sample["input value"]):
                    already_sus = True
            if not already_sus:
                mislabels.append(new_sample)
    return mislabels, wrong_samples


def run_pipeline(model_adapter: BaseModelAdapter, data: list[np.ndarray]):

    training_settings = model_adapter.config.training_settings

    scores = []
    class_acc = []

    for R in range(training_settings.R):
        print(f"Now running R: {R+1}/{training_settings.R}")
        score, acc = kCV_outer(model_adapter, data, R)
        scores.append(score)
        class_acc.append(acc)
    scores = np.concatenate(scores, axis=0)
    class_acc = np.concatenate(class_acc, axis=0)

    np.save(model_adapter.output_dir, scores)
    np.save(model_adapter.output_dir + "_fold_acc", class_acc)


def run_defect_pipeline(model_adapter: BaseModelAdapter, data: list[np.ndarray], size):
    training_settings = model_adapter.config.training_settings
    scores = []
    class_acc = []
    seed = [42, 2, 4, 7, 6, 9, 94, 23, 62, 65]
    index = [
        7,
        37,
        100,
        141,
        158,
        179,
        210,
        228,
        276,
        308,
        410,
        427,
        519,
        550,
        558,
        597,
        752,
        762,
        796,
        809,
    ]

    inputs = pd.read_csv("./data/X_TR.csv").to_numpy()
    inputs = np.delete(inputs, index, axis=1)

    for R in range(training_settings.R):
        print(f"Now running R: {R+1}/{training_settings.R}")
        rng = np.random.default_rng(seed=seed[R])
        idx = rng.choice(np.arange(inputs.shape[-1]), size=size, replace=False)
        added_features = inputs[:, idx]
        small_input, labels = data
        disrupted_input = np.concatenate([small_input, added_features], axis=1)
        data_disrupt = [disrupted_input, labels]
        model_adapter.config.hyperparameters["in_features"] = disrupted_input.shape[-1]

        score, acc = kCV_outer(model_adapter, data_disrupt, R)
        scores.append(score)
        class_acc.append(acc)
    scores = np.concatenate(scores, axis=0)
    class_acc = np.concatenate(class_acc, axis=0)

    np.save(model_adapter.output_dir, scores)
    np.save(model_adapter.output_dir + "_fold_acc", class_acc)


def main():
    space = [Real(0.1, 0.2, name="lr")]
    opt = Optimizer(dimensions=space)
    next_config = opt.ask()
    opt.tell(next_config, -0.3)
    next_config = opt.ask()

    opt.tell(next_config, -0.2)
    print(opt.get_result())


if __name__ == "__main__":
    main()
