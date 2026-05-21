import numpy as np
from sklearn.model_selection import StratifiedKFold
from dataclasses import dataclass, field
from typing import Any, Dict, Type, Optional, List
from skopt import Optimizer
from skopt.space import Integer, Real, Categorical


@dataclass
class BaseTrainConfig:
    search_space: List[Any] = field(default_factory=list)
    output_dir: str = ".models/"
    seed: list = [6, 7, 42, 67, 69]
    R: int = 5
    K: int = 10
    L: int = 5


@dataclass
class ModelConfig:
    name: "str"
    model_class: Type[Any]
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_settings: BaseTrainConfig = field(default_factory=BaseTrainConfig)


class BaseModelAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.model = config.model_class(**config.hyperparameters)

    def clean_model(self):
        self.model = self.config.model_class(**self.config.hyperparameters)

    def train_params(self, train_batch):
        raise NotImplementedError

    def validate(self, val_batch) -> float:
        raise NotImplementedError


def kCV_inner(
    model_adapter: BaseModelAdapter, hyper_opt_data, outer_fold: Optional[int] = None
) -> np.float32:
    training_settings = model_adapter.config.training_settings

    skf = StratifiedKFold(n_splits=training_settings.L, shuffle=True)

    if outer_fold:
        skf.random_state = training_settings.seed[outer_fold]

    best_fold_score = 0.0
    fold_score = np.zeros(training_settings.L)

    for fold, (train_idx, val_idx) in enumerate(skf.split(*hyper_opt_data)):
        print(f"Fold: {fold +1 }/{training_settings.L}")
        train_batch = [values[train_idx] for values in hyper_opt_data]
        val_batch = [values[val_idx] for values in hyper_opt_data]

        model_adapter.clean_model()

        model_adapter.train_params(train_batch)
        score = model_adapter.validate(val_batch)

        if score > best_fold_score:
            best_fold_score = score

        fold_score[fold] = score
        print(f"Best val score: {score}")
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
        next_config = opt.ask()

        if isinstance(next_config, list):
            for dimension, val in zip(training_settings.search_space, next_config):
                model_adapter.config.hyperparameters[dimension.name] = val
        else:
            raise TypeError(
                f"Unexpected return type from opt.ask(): {type(next_config)}"
            )
        mean_score = kCV_inner(model_adapter, hyper_opt_data)
        cost = -mean_score
        opt.tell(next_config, cost)
    results = opt.get_result()


def kCV_outer():
    raise NotImplementedError
