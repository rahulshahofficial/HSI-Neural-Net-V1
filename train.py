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
    """
    Trainer class for the hyperspectral reconstruction model.

    This class handles the complete training pipeline including:
    - Training loop management
    - Validation
    - Loss calculation and optimization
    - Model saving
    - Performance visualization

    The trainer is designed to work with the SRNet architecture which requires
    both filtered measurements and filter pattern tensors as input.
    """
    def __init__(self, model, train_loader, val_loader=None):
        """
        Initialize the training environment.

        Args:
            model: The neural network model (SRNet instance)
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
        """
        # Determine whether to use GPU or CPU based on availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training will use: {self.device}")

        # Move model to the selected device
        self.model = model.to(self.device)

        # Store data loaders
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Initialize loss function (Mean Squared Error for reconstruction tasks)
        self.criterion = nn.MSELoss()

        # Initialize optimizer (Adam with learning rate from config)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)

        # Initialize metrics tracking
        self.train_losses = []  # Track training loss history
        self.val_losses = []    # Track validation loss history
        self.best_val_loss = float('inf')  # Track best validation performance

    def train_epoch(self):
        """
        Train the model for one complete epoch.

        An epoch consists of iterating through all training batches once.
        For each batch, the function:
        1. Moves data to the appropriate device
        2. Performs a forward pass through the model
        3. Calculates the loss
        4. Performs backpropagation
        5. Updates model weights

        Returns:
            Average loss for the epoch
        """
        # Set model to training mode (enables dropout, batch norm updates, etc.)
        self.model.train()

        # Initialize running loss and batch counter
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        # Iterate through data batches
        for batch_idx, (filtered_measurements, filter_pattern, spectra) in enumerate(self.train_loader):
            # Move data to device (GPU/CPU)
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)
            spectra = spectra.to(self.device)

            # Reset gradients before forward pass
            self.optimizer.zero_grad()

            # Forward pass: Get model predictions
            # Note: SRNet takes both measurements and filter pattern as input
            outputs = self.model(filtered_measurements, filter_pattern)

            # Calculate loss using model's compute_loss function
            # This includes reconstruction, spectral smoothness, and spatial consistency losses
            loss = self.model.compute_loss(outputs, spectra, self.criterion)

            # Backward pass: Calculate gradients
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Accumulate loss for reporting
            epoch_loss += loss.item()

            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f'Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.6f}')

        # Return average loss for the epoch
        return epoch_loss / num_batches

    def validate(self):
        """
        Evaluate the model on validation data.

        This function:
        1. Sets the model to evaluation mode
        2. Performs forward passes without gradient calculation
        3. Calculates validation loss

        Returns:
            Average validation loss, or None if no validation data is provided
        """
        # Skip validation if no validation data is provided
        if self.val_loader is None:
            return None

        # Set model to evaluation mode (disables dropout, etc.)
        self.model.eval()
        val_loss = 0.0

        # Disable gradient calculation for efficiency and to prevent memory leaks
        with torch.no_grad():
            for filtered_measurements, filter_pattern, spectra in self.val_loader:
                # Move data to device
                filtered_measurements = filtered_measurements.to(self.device)
                filter_pattern = filter_pattern.to(self.device)
                spectra = spectra.to(self.device)

                # Forward pass
                outputs = self.model(filtered_measurements, filter_pattern)

                # Calculate loss using the same function as training
                loss = self.model.compute_loss(outputs, spectra, self.criterion)
                val_loss += loss.item()

        # Return average validation loss
        return val_loss / len(self.val_loader)

    def train(self, num_epochs=None, resume_from=None):
        """
        Execute the complete training process for multiple epochs.

        This function:
        1. Runs training for the specified number of epochs
        2. Performs validation after each epoch if validation data is available
        3. Saves the best model based on validation performance
        4. Tracks and plots losses

        Args:
            num_epochs: Number of epochs to train for (default: from config)
            resume_from: Path to checkpoint to resume training from (not implemented yet)
        """
        # Use config value if num_epochs not specified
        if num_epochs is None:
            num_epochs = config.num_epochs

        print(f"Training on {self.device}")
        print(f"Training with {len(self.train_loader)} batches per epoch")

        # Main training loop
        for epoch in range(num_epochs):
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

                # Save best model based on validation performance
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
                    torch.save(self.model.state_dict(), config.model_save_path)
                    print(f"Model saved to {config.model_save_path} (new best: {val_loss:.6f})")

        print("Training completed!")

        # Plot loss curves
        self.plot_losses()

        # Save final model if no validation was performed
        if self.val_loader is None:
            os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
            torch.save(self.model.state_dict(), config.model_save_path)
            print(f"Final model saved to {config.model_save_path}")

    def evaluate_model(self, test_loader):
        """
        Evaluate the trained model on test data.

        This function:
        1. Sets the model to evaluation mode
        2. Performs forward passes on test data
        3. Calculates test loss
        4. Collects model outputs for further analysis

        Args:
            test_loader: DataLoader for test data

        Returns:
            tuple: (average_test_loss, all_outputs, all_targets)
                - average_test_loss: Mean loss on test data
                - all_outputs: List of model predictions
                - all_targets: List of ground truth values
        """
        # Set model to evaluation mode
        self.model.eval()
        test_loss = 0.0
        all_outputs = []
        all_targets = []

        # Disable gradient calculation
        with torch.no_grad():
            for filtered_measurements, filter_pattern, spectra in test_loader:
                # Move data to device
                filtered_measurements = filtered_measurements.to(self.device)
                filter_pattern = filter_pattern.to(self.device)
                spectra = spectra.to(self.device)

                # Forward pass
                outputs = self.model(filtered_measurements, filter_pattern)

                # Calculate loss (using simple MSE for evaluation)
                loss = self.criterion(outputs, spectra)
                test_loss += loss.item()

                # Store outputs and targets for analysis
                all_outputs.append(outputs.cpu())
                all_targets.append(spectra.cpu())

        # Calculate and report average loss
        avg_test_loss = test_loss / len(test_loader)
        print(f"Average Test Loss: {avg_test_loss:.6f}")

        return avg_test_loss, all_outputs, all_targets

    def plot_losses(self):
        """
        Visualize training and validation losses over epochs.

        This function creates and saves a plot showing:
        1. Training loss curve
        2. Validation loss curve (if available)
        """
        plt.figure(figsize=(10, 6))

        # Plot training loss
        plt.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)

        # Plot validation loss if available
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)

        # Add plot details
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training and Validation Losses', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Add markers at lowest points
        if self.train_losses:
            min_train_idx = np.argmin(self.train_losses)
            min_train_loss = self.train_losses[min_train_idx]
            plt.plot(min_train_idx, min_train_loss, 'bo', markersize=8)
            plt.annotate(f'Min: {min_train_loss:.4f}',
                        (min_train_idx, min_train_loss),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        if self.val_losses:
            min_val_idx = np.argmin(self.val_losses)
            min_val_loss = self.val_losses[min_val_idx]
            plt.plot(min_val_idx, min_val_loss, 'ro', markersize=8)
            plt.annotate(f'Min: {min_val_loss:.4f}',
                        (min_val_idx, min_val_loss),
                        textcoords="offset points",
                        xytext=(0,10),
                        ha='center')

        # Save plot
        os.makedirs('plots', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'plots/loss_plot_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss plot saved to plots/loss_plot_{timestamp}.png")
