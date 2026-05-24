from train_pipeline import BaseTrainConfig, BaseModelAdapter
from dataclasses import dataclass
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
from torch.utils.data import DataLoader
from data_processing import AnimalPictures


@dataclass
class TorchTrainConfig(BaseTrainConfig):
    max_epochs = 100
    batch_train = 32
    batch_val = 64
    train_val_split = 0.8
    lr = 0.01
    optimizer = AdamW
    scheduler = CosineAnnealingLR
    loss_function = nn.BCEWithLogitsLoss()


class LogRegAdapter(BaseModelAdapter[TorchTrainConfig]):
    def train_params(self, train_batch: list):
        train_cfg = self.config.training_settings

        images, labels = train_batch
        images = self.dimred.fit_transform(images, y=labels)

        sample_size = images.shape[0]
        train_split = int(sample_size * train_cfg.train_val_split)

        train_images = images[:train_split]
        train_labels = labels[:train_split]
        val_images = images[train_split:]
        val_labels = labels[train_split:]

        dataset_train = AnimalPictures(train_images, train_labels)
        dataset_val = AnimalPictures(val_images, val_labels)

        train_loader = DataLoader(
            dataset_train,
            batch_size=train_cfg.batch_train,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset_val,
            batch_size=train_cfg.batch_val,
            shuffle=False,
        )
        opt = train_cfg.optimizer(
            self.model.parameters(),
            lr=train_cfg.lr,
        )
        scheduler = train_cfg.scheduler(opt, train_cfg.max_epochs)

        self.model.train_params(
            train_cfg.max_epochs,
            train_loader,
            val_loader,
            train_cfg.loss_function,
            opt,
            scheduler,
        )

    def validate(self, val_batch):
        self.model.eval()
        val_images, val_labels = val_batch
        val_images = self.dimred.transform(val_images)

        with torch.no_grad():
            logits = self.model(val_images)
            probs = torch.sigmoid(logits)
            preds = (probs > self.model.threshold).int()

        return preds, val_labels


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, threshold=0.5):
        super().__init__()
        self.input_dim = input_dim
        self.layer = nn.Linear(input_dim, 1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

    def __str__(self):
        return "logreg"

    def forward(self, x):
        output = self.layer(x)
        return output

    def validate_model(self, val_loader: DataLoader, loss_function) -> float:
        self.eval()
        total_loss = 0.0
        total_batches = len(val_loader)

        with torch.no_grad():
            for val_input, val_target in val_loader:
                val_input, val_target = (
                    val_input.to(self.device),
                    val_target,
                )

                logits = self(val_input).cpu()
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
        stopper=3,
    ):

        max_loss = np.inf
        epochs_getting_worse = 0
        best_model = None

        print(f"training running on {self.device}")
        self.to(self.device)

        for epoch in range(max_epochs):
            print(f"Epoch: {epoch+1}")

            avg_loss = self.train_epoch(train_loader, loss_function, optimizer)
            avg_val_loss = self.validate_model(val_loader, loss_function)

            print(f"Training loss: {avg_loss:.4f}, Validation loss: {avg_val_loss:.4f}")
            scheduler.step(avg_val_loss)

            if avg_val_loss < max_loss:
                epochs_getting_worse = 0
                max_loss = avg_val_loss
                best_model = self.state_dict
            else:
                epochs_getting_worse += 1
            if epochs_getting_worse >= stopper:
                break

        return best_model
