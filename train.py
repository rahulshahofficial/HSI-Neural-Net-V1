import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

from config import config

class Trainer:
    def __init__(self, model, train_loader, val_loader=None):
        """
        Initialize the trainer.
        Args:
            model: The neural network model
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize loss function and optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Track losses per epoch
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')

    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (filtered_measurements, spectra) in enumerate(self.train_loader):
            # Move data to device
            filtered_measurements = filtered_measurements.to(self.device)
            spectra = spectra.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(filtered_measurements)
            loss = self.criterion(outputs, spectra)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Accumulate loss
            epoch_loss += loss.item()

            # Print progress
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.6f}')

        return epoch_loss / num_batches

    def validate(self):
        """Perform validation."""
        if self.val_loader is None:
            return None

        self.model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for filtered_measurements, spectra in self.val_loader:
                filtered_measurements = filtered_measurements.to(self.device)
                spectra = spectra.to(self.device)

                outputs = self.model(filtered_measurements)
                loss = self.criterion(outputs, spectra)
                val_loss += loss.item()

        return val_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }

        # Save regular checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        # Save best model if this is the best validation loss
        if is_best:
            best_model_path = 'checkpoints/best_model.pth'
            torch.save(checkpoint, best_model_path)
            print(f"New best model saved with validation loss: {val_loss:.6f}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        return start_epoch

    def plot_losses(self):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss')
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss')

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)

        # Save plot
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots/loss_plot_{timestamp}.png')
        plt.close()

    def train(self, num_epochs=None, resume_from=None):
        """
        Full training loop.
        Args:
            num_epochs: Number of epochs to train for (default: from config)
            resume_from: Path to checkpoint to resume training from
        """
        if num_epochs is None:
            num_epochs = config.num_epochs

        start_epoch = 0
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch + 1}")

        print(f"Training on {self.device}")
        print(f"Training with {len(self.train_loader)} batches per epoch")

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            # Training phase
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            print(f"Training Loss: {train_loss:.6f}")

            # Validation phase
            if self.val_loader is not None:
                val_loss = self.validate()
                self.val_losses.append(val_loss)
                print(f"Validation Loss: {val_loss:.6f}")

                # Save checkpoint if this is the best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch + 1, val_loss, is_best=True)

            # Save regular checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch + 1, val_loss if self.val_loader else train_loss)

        print("Training completed!")
        self.plot_losses()

        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        torch.save(self.model.state_dict(), config.model_save_path)
        print(f"Model saved to {config.model_save_path}")

    def evaluate_model(self, test_loader):
        """
        Evaluate model on test data.
        Args:
            test_loader: DataLoader for test data
        Returns:
            Average test loss and reconstructed spectra
        """
        self.model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []

        with torch.no_grad():
            for filtered_measurements, spectra in test_loader:
                filtered_measurements = filtered_measurements.to(self.device)
                spectra = spectra.to(self.device)

                outputs = self.model(filtered_measurements)
                loss = self.criterion(outputs, spectra)
                test_loss += loss.item()

                # Store outputs and targets for analysis
                all_outputs.append(outputs.cpu())
                all_targets.append(spectra.cpu())

        avg_test_loss = test_loss / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss:.6f}")

        return avg_test_loss, all_outputs, all_targets
