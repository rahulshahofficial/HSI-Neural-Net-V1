import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime

from config import config
from dataset import HyperspectralDataset
from network import HyperspectralNet

class Trainer:
    def __init__(self, model, train_loader, val_loader=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0

        for filtered_measurements, spectra in self.train_loader:
            filtered_measurements = filtered_measurements.to(self.device)
            spectra = spectra.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(filtered_measurements)
            loss = self.criterion(outputs, spectra)
            loss.backward()

            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for filtered_measurements, spectra in self.val_loader:
                filtered_measurements = filtered_measurements.to(self.device)
                spectra = spectra.to(self.device)

                outputs = self.model(filtered_measurements)
                loss = self.criterion(outputs, spectra)
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def train(self, num_epochs=None):
        if num_epochs is None:
            num_epochs = config.num_epochs

        print(f"Starting training on {self.device}")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)

            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}]')
                print(f'Training Loss: {train_loss:.6f}')
                if self.val_loader is not None:
                    print(f'Validation Loss: {val_loss:.6f}')

        self.save_model()
        self.plot_training_history()

    def save_model(self):
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), config.model_save_path)
        print(f"Model saved to {config.model_save_path}")

    def plot_training_history(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True)

        os.makedirs(config.results_path, exist_ok=True)
        plt.savefig(os.path.join(
            config.results_path,
            f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        ))
        plt.close()
