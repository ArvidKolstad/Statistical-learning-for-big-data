import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torch.optim import Adam
from dim_red import dimension_reduction
from sklearn.model_selection import StratifiedKFold


class ReducedDimDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = torch.tensor(inputs)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class MultilayerPerception(nn.Module):
    def __init__(self, layer_dim, act_func, dropout_rate=0.3):
        super().__init__()
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
        save_model=False,
    ):

        self.to(self.device)
        params = self.state_dict()
        best_val_loss = np.inf

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for inputs, labels in train_data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                running_loss += loss.item()
            avg_t_loss = running_loss / len(train_data_loader)
            avg_v_loss = self.validate_model(val_data_loader, loss_function)
            scheduler.step(avg_v_loss)
            if best_val_loss > avg_v_loss:
                best_val_loss = avg_v_loss
                if save_model:
                    params = self.state_dict()
            print(
                f"Epoch {epoch +1}/{epochs}, Training loss: {avg_t_loss:.5f}, Validation loss: {avg_v_loss:.5f}"
            )

        if save_model:
            torch.save(params, "./saved_models/mlp")
        return best_val_loss

    def validate_model(self, val_loader, loss_function):
        self.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                loss = loss_function(outputs.squeeze(1), labels)
                val_loss += loss.item()
        avg_loss = val_loss / len(val_loader)
        return avg_loss


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
):

    skf = StratifiedKFold(n_splits=k, shuffle=True)
    fold_score = []
    matrixes = [data[0] for data in data_set]
    labels = [data[1] for data in data_set]

    for fold, (train_idx, val_idx) in enumerate(skf.split(matrixes, labels)):
        print(f"Fold: {fold}")
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
        fold_score.append(score)
    return np.mean(fold_score)


# tänkte att man kunde göra en funktion för hyperparameteroptimeringen eftersom de är i olika dicts så kan man säga vilken dict de befinner sig genom get_dict och sen använda den i hyper_parameter_opt för att skriva mindre kod.
def get_dict(module):
    if module == "train":
        return 0
    elif module == "optimizer":
        return 1
    elif module == "scheduler":
        return 2
    else:
        raise ValueError("Module doesn't exist")


def hyper_parameter_opt(hyper_parameter, parameter_range, module, params):
    print("hi")


def main():
    input_layer = 3
    classes = 10
    data_batches = 10

    layout = {
        "layer_dim": [input_layer, 256, 128, 128, 64, 64, 32, classes],
        "act_func": ["ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "ReLU", "identity"],
        "dropout_rate": 0.2,
    }
    optimizer = Adam
    optimizer_settings = {
        "params": None,
        "lr": 0.001,
        "weight_decay": 1e-4,
    }
    scheduler = ReduceLROnPlateau
    scheduler_settings = {"optimizer": None, "patience": 5, "factor": 0.5}

    training_matrix = np.load("./data/train_matrix.npy")
    training_labels = np.load("./data/train_labels.npy")
    red_training_matrix = dimension_reduction(
        training_matrix, train_label=training_labels, n_dimensions=input_layer
    )
    train_settings = {
        "epochs": 100,
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
    score = kCV(
        data_batches,
        MultilayerPerception,
        data_set,
        layout,
        train_settings,
        optimizer,
        optimizer_settings,
        scheduler,
        scheduler_settings,
    )
    print(score)


if __name__ == "__main__":
    main()
