import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import pickle as pkl
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim import Adam
from dim_red import dimension_reduction
from sklearn.model_selection import StratifiedKFold


class ReducedDimDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.labels = torch.tensor(labels).long()

        self.inputs = (self.inputs - self.inputs.mean()) / (self.inputs.std() + 1e-8)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class MultilayerPerception(nn.Module):
    def __init__(self, layer_dim, act_func, dropout_rate=0.3):
        super().__init__()

        self.layer_dim = layer_dim
        self.act_func = act_func
        self.dropout_rate = dropout_rate

        self.layer_count = len(layer_dim)
        self.act_func_length = len(act_func)

        assert self.layer_count == self.act_func_length + 1

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        for i in range(self.layer_count - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(get_activation_function(act_func[i]))
            if i < self.layer_count - 2:
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

        self._is_trained = False

    def forward(self, x):
        return self.network(x)

    def train_params(
        self,
        epochs,
        train_data_loader,
        val_data_loader,
        loss_function,
        optimizer,
        scheduler,
    ):

        self.to(self.device)

        best_val_acc = 0.0
        best_state = self.state_dict()

        for epoch in range(epochs):
            total = 0
            correct = 0

            self.train()
            running_loss = 0.0

            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad(set_to_none=True)

                outputs = self(inputs)
                loss = loss_function(outputs, labels)

                loss.backward()
                optimizer.step()  # La till detta

                running_loss += loss.item()

                # Nytt
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            train_acc = correct / total

            val_acc = self.validate_model(val_data_loader)

            scheduler.step(
                1 - val_acc
            )  # ReduceLROnPlateau (högre är bättre → invertera)

            if best_val_acc < val_acc:
                best_val_acc = val_acc
                best_state = self.state_dict()

            if (epoch + 1) % 100 == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train acc: {train_acc:.4f} | "
                    f"Val acc: {val_acc:.4f}"
                )

        self.load_state_dict(best_state)
        return best_val_acc

    def predict(self, X):
        self.eval()

        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        with torch.no_grad():
            outputs = self(X)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def validate_model(self, val_loader):
        self.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self(inputs)

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return accuracy

    def save_model_settings(self, model_name: str):
        settings = {
            "layer_dim": self.layer_dim,
            "act_func": self.act_func,
            "dropout_rate": self.dropout_rate,
        }
        with open("./saved_models/mlp_settings_" + model_name, "wb") as f:
            pkl.dump(settings, f)


def load_mlp_model(mlp_settings_path, mlp_params_path):
    with open(mlp_settings_path, "rb") as f:
        settings = pkl.load(f)

    model = MultilayerPerception(**settings)

    state_dict = torch.load(
        mlp_params_path,
        map_location=torch.device("cpu"),
        weights_only=True
    )

    # model.load_state_dict(torch.load(mlp_params_path, weights_only=True))
    model.load_state_dict(state_dict)

    model.eval()
    return model


def get_activation_function(activation_name):
    if activation_name == "ReLU":
        return nn.ReLU()
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif activation_name == "tanh":
        return nn.Tanh()
    elif activation_name == "softmax":
        return nn.Softmax()
    elif activation_name == "logsoftmax":
        return nn.LogSoftmax()
    elif activation_name == "identity":
        return nn.Identity()
    else:
        raise ValueError("Activation name is wrong or not implemented")


def kCV(
    k,
    model_class,
    data_set,
    network_layout,
    training_settings,
    optimizer,
    optimizer_settings,
    scheduler,
    scheduler_settings,
    save_model=False,
    model_name="normal",
):

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    best_fold_score = 0.0
    fold_score = []

    matrixes = data_set.inputs.detach().cpu().numpy()
    labels = data_set.labels.detach().cpu().numpy()

    for fold, (train_idx, val_idx) in enumerate(skf.split(matrixes, labels)):
        print(f"Fold: {fold +1 }/{k}")

        train_data_loader = DataLoader(
            data_set, batch_size=100, sampler=SubsetRandomSampler(train_idx)
        )

        val_data_loader = DataLoader(
            data_set, batch_size=256, sampler=SubsetRandomSampler(val_idx)
        )

        training_settings["train_data_loader"] = train_data_loader
        training_settings["val_data_loader"] = val_data_loader

        model = model_class(**network_layout)
        model.to(model.device)

        optimizer_settings["params"] = model.parameters()
        opt_func = optimizer(**optimizer_settings)

        scheduler_settings["optimizer"] = opt_func
        sched_func = scheduler(**scheduler_settings)

        training_settings["optimizer"] = opt_func
        training_settings["scheduler"] = sched_func

        score = model.train_params(**training_settings)
        if score > best_fold_score and save_model:
            best_fold_score = score

            torch.save(model.state_dict(), "./saved_models/mlp_" + model_name)
            model.save_model_settings(model_name)

        fold_score.append(score)
        print(f"Best val score: {score}")
    return fold_score


def get_dict(module):
    if module == "data":
        return 2
    elif module == "network":
        return 3
    elif module == "train":
        return 4
    elif module == "optimizer":
        return 6
    elif module == "scheduler":
        return 8
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
        if redo_data:
            if hyper_parameter == "layer_dim":
                red_training_matrix, _ = dimension_reduction(
                    data_matrix, n_dim_pca=value
                )
                input_dim = red_training_matrix.shape[1]
                new_layout = original_value.copy()
                new_layout[0] = input_dim
                value = new_layout
            else:
                red_training_matrix, _ = dimension_reduction(data_matrix, n_dim_pca=0.8)

            data_set_train = ReducedDimDataset(
                red_training_matrix,
                data_label,
            )
            params[get_dict("data")] = data_set_train

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


def main():
    training_matrix = np.load("./data/train_matrix.npy")
    training_labels = np.load("./data/train_labels.npy")

    red_training_matrix, _ = dimension_reduction(
        training_matrix, train_label=training_labels, n_dim_pca=44
    )

    input_layer = red_training_matrix.shape[1]
    classes = 10
    data_batches = 10

    layout = {
        "layer_dim": [input_layer, 256, 128, 64, 32, classes],
        "act_func": ["ReLU", "ReLU", "ReLU", "ReLU", "identity"],
        "dropout_rate": 0.3,
    }

    optimizer = Adam
    optimizer_settings = {
        "params": None,
        "lr": 0.005,
        "weight_decay": 0.001,
    }

    scheduler = ReduceLROnPlateau
    scheduler_settings = {"optimizer": None, "patience": 5, "factor": 0.5}

    train_settings = {
        "epochs": 500,
        "train_data_loader": None,
        "val_data_loader": None,
        "loss_function": nn.CrossEntropyLoss(),
        "optimizer": None,
        "scheduler": None,
    }

    data_set = ReducedDimDataset(
        red_training_matrix,
        training_labels,
    )

    params = {
        "k": data_batches,
        "model_class": MultilayerPerception,
        "data_set": data_set,
        "network_layout": layout,
        "training_settings": train_settings,
        "optimizer": optimizer,
        "optimizer_settings": optimizer_settings,
        "scheduler": scheduler,
        "scheduler_settings": scheduler_settings,
        "save_model": True,
        "model_name": None,
    }

    # kCV(**params)

    # part 2
    misslabeled_data = [
        "./data/train_labels_0.1_mislabel.npy",
        "./data/train_labels_0.1_mislabel.npy",
        "./data/train_labels_0.1_mislabel.npy",
    ]
    model_name = ["light", "moderate", "heavy"]
    for idx, path in enumerate(misslabeled_data):
        bad_label = np.load(path)
        bad_data = ReducedDimDataset(red_training_matrix, bad_label)
        params["data_set"] = bad_data
        params["model_name"] = model_name[idx]
        kCV(**params)

    # hyper parameter optimization
    """
    hyper_parameter_opt(
        "layer_dim",
        [0.99, 0.90, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        "network",
        params,
        "normal",
        data_matrix=training_matrix,
        data_label=training_labels,
    )

    hyper_parameter_opt(
        "dropout_rate",
        [0.0, 0.05, 0.10, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.45, 0.5],
        "network",
        params,
        "normal",
    )
    hyper_parameter_opt(
        "weight_decay",
        [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3],
        "optimizer",
        params,
        "normal",
    )
    hyper_parameter_opt(
        "lr",
        [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        "optimizer",
        params,
        "normal",
    )

"""


if __name__ == "__main__":
    main()
