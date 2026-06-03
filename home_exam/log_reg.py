from train_pipeline import BaseTrainConfig, BaseModelAdapter
from dataclasses import dataclass, field
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from data_process import B11_dataset
from torch.utils.data import DataLoader


@dataclass
class TorchTrainConfig(BaseTrainConfig):
    class_imbalance: torch.Tensor = field(
        default_factory=lambda: torch.tensor(np.ones(7) / 7)
    )
    max_epochs = 100
    batch_train = 32
    batch_val = 64
    train_val_split = 0.8
    lr = 0.01
    optimizer = AdamW
    scheduler = CosineAnnealingLR
    loss_function = nn.CrossEntropyLoss


class LogRegAdapter(BaseModelAdapter[TorchTrainConfig]):
    def train_params(self, train_batch: list):
        train_cfg = self.config.training_settings

        images, labels = train_batch

        sample_size = images.shape[0]
        train_split = int(sample_size * train_cfg.train_val_split)

        train_images = torch.tensor(images[:train_split]).float()
        train_labels = torch.tensor(labels[:train_split]).long()
        val_images = torch.tensor(images[train_split:]).float()
        val_labels = torch.tensor(labels[train_split:]).long()

        dataset_train = B11_dataset(train_images, train_labels)
        dataset_val = B11_dataset(val_images, val_labels)

        train_loader = DataLoader(
            dataset_train,
            batch_size=train_cfg.batch_train,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=train_cfg.batch_val,
            shuffle=False,
            num_workers=0,
        )
        opt = train_cfg.optimizer(
            self.model.parameters(), lr=train_cfg.lr, weight_decay=self.model.l2
        )
        scheduler = train_cfg.scheduler(opt, train_cfg.max_epochs)

        best_model = self.model.train_params(
            train_cfg.max_epochs,
            train_loader,
            val_loader,
            train_cfg.loss_function(weight=train_cfg.class_imbalance.to("cuda")),
            opt,
            scheduler,
        )
        self.model.load_state_dict(best_model)

    def validate(self, val_batch):
        self.model.eval()
        self.model.to(self.model.device)
        val_images, val_labels = val_batch
        val_images = torch.tensor(val_images).float().to(self.model.device)

        with torch.no_grad():
            logits = self.model(val_images).detach().cpu()
            probs = torch.softmax(logits, dim=-1)
            preds = np.argmax(probs, axis=-1).numpy()

        return preds, val_labels

    def get_probability(self, val_input) -> np.ndarray:
        return self.model.get_probabilities(val_input)

    def save_model(self):
        self.model.save(self.output_dir)

    def load_model(self):
        self.model.load(self.output_dir)


class LogisticRegression(nn.Module):
    def __init__(self, in_features, number_of_classes, l2=0.001):
        super().__init__()
        self.in_features = in_features
        self.layer = nn.Linear(in_features, number_of_classes)
        self.l2 = l2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __str__(self):
        return "logreg"

    def forward(self, x):
        output = self.layer(x)
        return output

    def get_probabilities(self, inputs) -> np.ndarray:
        with torch.no_grad():
            inputs = torch.tensor(inputs).float()
            probability = torch.softmax(self(inputs), dim=-1).numpy()
            return probability

    def validate_model(self, val_loader: DataLoader, loss_function) -> float:
        self.eval()
        total_loss = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_target in val_loader:
                val_input, val_target = (
                    val_input.to(self.device),
                    val_target.to(self.device),
                )
                logits = self(val_input)
                loss = loss_function(logits, val_target)

                total_loss += loss.item()

        mean_val_loss = total_loss / total_batches

        return mean_val_loss

    def train_epoch(self, train_loader, loss_function, optimizer):
        self.train()
        total_loss = 0.0
        total_batches = len(train_loader)
        for train_input, train_labels in train_loader:

            train_input, train_labels = (
                train_input.to(self.device),
                train_labels.to(self.device),
            )
            optimizer.zero_grad()
            logits = self(train_input)
            loss = loss_function(logits, train_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            optimizer.step()
        mean_loss = total_loss / total_batches

        return mean_loss

    def train_params(
        self,
        max_epochs,
        train_loader,
        val_loader,
        loss_function,
        optimizer,
        scheduler,
        stopper=4,
    ):

        max_loss = np.inf
        epochs_getting_worse = 0
        best_model = None

        self.to(self.device)

        for _ in range(max_epochs):

            self.train_epoch(train_loader, loss_function, optimizer)

            avg_val_loss = self.validate_model(val_loader, loss_function)

            scheduler.step()

            if avg_val_loss < max_loss:
                epochs_getting_worse = 0
                max_loss = avg_val_loss
                best_model = self.state_dict()
            else:
                epochs_getting_worse += 1
            if epochs_getting_worse >= stopper:
                break

        return best_model

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        self.eval()
