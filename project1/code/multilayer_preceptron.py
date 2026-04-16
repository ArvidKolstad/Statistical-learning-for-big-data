import numpy as np
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torch.optim import SGD
from dim_red import dimension_reduction


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
        self, epochs, data_loader, loss_function, optimizer, scheduler, save_model=False
    ):

        self.to(self.device)
        params = self.state_dict()
        best_val_loss = np.inf

        for epoch in range(epochs):
            self.train()
            running_loss = 0.0

            for inputs, labels in data_loader:
                optimizer.zero_grad(set_to_none=True)
                outputs = self(inputs)
                loss = loss_function(outputs, labels)
                loss.backwards()
                running_loss += loss.item()
            avg_t_loss = running_loss / len(data_loader)
            avg_v_loss = self.validate_model(self, loss_function)
            scheduler.step(avg_v_loss)
            if best_val_loss < avg_v_loss:
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


def main():
    input_layer = 3
    classes = 10
    settings = {
        "layer_dim": [input_layer, 40, 40, classes],
        "act_func": ["ReLU", "ReLU", "identity"],
        "dropout_rate": 0.2,
    }
    loss_function = nn.CrossEntropyLoss()
    model = MultilayerPerception(**settings)
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    training_matrix = np.load("./data/train_matrix.npy")
    training_labels = np.load("./data/train_labels.npy")
    red_training_matrix = dimension_reduction(
        training_matrix, train_label=training_labels
    )

    data_set = ReducedDimDataset(red_training_matrix, training_labels)
    data_loader = DataLoader(data_set)
    epochs = 20

    model.train_params(epochs, data_loader, loss_function, optimizer, scheduler)


if __name__ == "__main__":
    main()
