import torch.nn as nn
from train_utils import kCV
import numpy as np
import torch


class MultilayerPerception(nn.Module):
    def __init__(self, layer_dim, act_func, dropout_rate=0.3):
        super().__init__()
        self.layer_count = len(layer_dim)
        self.act_func_length = len(act_func)
        assert self.layer_count == self.act_func_length + 1

        layers = []
        for i in range(self.layer_count - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            layers.append(get_activation_function(act_func[i]))
            if i < self.layer_count - 2:
                layers.append(nn.Dropout(dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(
        self, convolution_layout, mlp_layout, activation_functions, dropout_rate=0.2
    ):
        super().__init__()
        conv_layers = []
        for idx in range(len(convolution_layout) - 1):
            conv_layers.append(
                nn.Conv2d(
                    convolution_layout[idx],
                    convolution_layout[idx + 1],
                    3,
                    padding="same",
                )
            )
            conv_layers.append(nn.BatchNorm2d(convolution_layout[idx + 1]))
            conv_layers.append(nn.ReLU())

            conv_layers.append(
                nn.Conv2d(
                    convolution_layout[idx + 1],
                    convolution_layout[idx + 1],
                    3,
                    padding="same",
                )
            )
            conv_layers.append(nn.BatchNorm2d(convolution_layout[idx + 1]))
            conv_layers.append(nn.ReLU())

            conv_layers.append(nn.MaxPool2d(2, 2))

        self.convolution_module = nn.Sequential(*conv_layers)

        self.connector = nn.AdaptiveAvgPool2d((2, 2))
        self.fc = MultilayerPerception(
            mlp_layout, activation_functions, dropout_rate=dropout_rate
        )

    def forward(self, inputs):
        x = inputs
        x = self.convolution_module(x)
        x = self.connector(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def run_training(
        self,
        epochs,
        loss_function,
        optimizer,
        train_data,
        val_data,
        file_name=None,
        scheduler=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        self.to(device)
        best_validation_loss = np.inf
        params = self.state_dict()

        for epoch in range(epochs):
            self.train()

            running_loss = 0.0
            for inputs, labels in train_data:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                outputs = self(inputs)

                loss = loss_function(outputs, labels)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            avg_t_loss = running_loss / len(train_data)
            avg_v_loss = self.validate_model(val_data, loss_function, device)
            if scheduler:
                scheduler.step(avg_v_loss)
            if avg_v_loss < best_validation_loss:
                best_validation_loss = avg_v_loss
                params = self.state_dict()
            print(
                f"Epoch {epoch +1}/{epochs}, Training loss: {avg_t_loss:.5f}, Validation loss: {avg_v_loss:.5f}"
            )

        if file_name:
            torch.save(params, "./saved_models/" + file_name)
        return best_validation_loss

    def validate_model(self, val_loader, loss_function, device):
        self.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
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
    input_size = 64 * 64
    folds = 8

    model_settings = {
        "convolution_layout": [64, 32, 16, 8],
        "mlp_layout": [8 * input_size, 256, 128, 64],
        "activation_functions": ["ReLU", "ReLU", "sigmoid"],
        "dropout_rate": 0.2,
    }
    train_matrix = np.load("./data/train_matrix.npy")
    train_labels = np.load("./data/train_labels.npy")

    settings = {
        "k": folds,
        "model_class": ConvolutionalNeuralNetwork,
        "train_input": train_matrix,
        "train_target": train_labels,
        "model_settings": model_settings,
        "training_settings":
    }
    kCV()

    model = ConvolutionalNeuralNetwork(**model_settings)


if __name__ == "__main__":
    main()
