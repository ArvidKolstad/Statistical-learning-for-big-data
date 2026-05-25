import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from dataclasses import dataclass, field
from typing import Any, Type, Optional, Generic, TypeVar
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical
from sklearn.decomposition import PCA


@dataclass
class BaseTrainConfig:
    search_space: list[Any]
    seed: list[int] = field(
        default_factory=lambda: [6, 7, 42, 10, 57, 67, 69, 103, 43, 37]
    )
    R: int = 10
    K: int = 5
    L: int = 3


T_Config = TypeVar("T_Config", bound=BaseTrainConfig)


@dataclass
class ModelConfig(Generic[T_Config]):
    name: "str"
    model_class: Type[Any]
    hyperparameters: dict[str, Any]
    training_settings: T_Config


class BaseModelAdapter(Generic[T_Config]):
    def __init__(self, config: ModelConfig[T_Config], save_path: str):
        self.config = config
        self.model = config.model_class(**config.hyperparameters)
        self.output_dir: str = save_path
        self.dimred = PCA(
            n_components=self.config.hyperparameters["in_features"],
            svd_solver="auto",
            random_state=42,
        )

    def clean_model(self):
        self.model = self.config.model_class(**self.config.hyperparameters)
        self.dimred = PCA(
            n_components=self.config.hyperparameters["in_features"],
            svd_solver="auto",
            random_state=42,
        )

    def train_params(self, train_batch) -> None:
        raise NotImplementedError

    def validate(self, val_batch) -> tuple[np.ndarray, np.ndarray]:
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
        print(f"Inner fold: {fold +1 }/{training_settings.L}")
        train_batch = [values[train_idx] for values in hyper_opt_data]
        val_batch = [values[val_idx] for values in hyper_opt_data]

        model_adapter.clean_model()

        model_adapter.train_params(train_batch)
        preds, labels = model_adapter.validate(val_batch)

        # accuracy
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
                print(f"Now testing for {dimension.name} = {val}")

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


def kCV_outer(
    model_adapter: BaseModelAdapter,
    data: list[np.ndarray],
    multiple_runs: Optional[int],
):
    training_settings = model_adapter.config.training_settings

    skf = StratifiedKFold(n_splits=training_settings.K, shuffle=True)

    if multiple_runs:
        skf.random_state = training_settings.seed[multiple_runs]

    fold_scores = np.zeros((training_settings.K, 3))

    for fold, (train_idx, val_idx) in enumerate(skf.split(*data)):
        print(f"Outer fold: {fold +1 }/{training_settings.K}")
        train_batch = [values[train_idx] for values in data]
        val_batch = [values[val_idx] for values in data]

        hyper_parameter_opt(model_adapter, train_batch, fold)
        model_adapter.clean_model()

        model_adapter.train_params(train_batch)

        preds, labels = model_adapter.validate(val_batch)

        correct_classification = np.sum(labels == preds)
        total_classification = labels.shape[0]

        accuracy = correct_classification / total_classification
        f1 = f1_score(labels, preds)
        auc = roc_auc_score(labels, preds)

        scores = np.array([accuracy, f1, auc])

        fold_scores[fold] = scores
    return fold_scores


def run_pipeline(model_adapter: BaseModelAdapter, data: list[np.ndarray]):
    training_settings = model_adapter.config.training_settings
    scores = []
    for R in range(training_settings.R):
        print(f"Now running R: {R+1}/{training_settings.R}")
        scores.append(kCV_outer(model_adapter, data, R))
    scores = np.concatenate(scores, axis=0)
    np.save(model_adapter.output_dir, scores)


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
