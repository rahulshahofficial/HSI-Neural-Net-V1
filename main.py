import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from config import config
from dataset import FullImageHyperspectralDataset
from network import FullImageHyperspectralNet
from train import Trainer

class HyperspectralProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    @staticmethod
    def get_all_files():
        """Get all available hyperspectral data files."""
        swir_path = os.path.join(config.dataset_path, "SWIR_RAW")
        return {
            'files': sorted([f.replace('.npy', '') for f in os.listdir(swir_path)
                           if f.endswith('.npy')]),
            'swir_path': swir_path
        }

    @staticmethod
    def load_data(num_images=None):
        """Load and preprocess hyperspectral data."""
        paths = HyperspectralProcessor.get_all_files()
        combined_data = []

        files_to_process = paths['files'][:num_images] if num_images else paths['files']

        print(f"Will process {len(files_to_process)} images")
        for filename in files_to_process:
            print(f"Loading file: {filename}")
            swir_file = os.path.join(paths['swir_path'], f"{filename}.npy")

            try:
                # Load and reshape SWIR data
                swir_data = np.load(swir_file)
                swir_data = swir_data.reshape((168, 211, 9))  # Known dimensions
                combined_data.append(swir_data)
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
                continue

        if not combined_data:
            raise ValueError("No data could be loaded")

        return np.stack(combined_data)

    def prepare_datasets(self, data):
        """Prepare training, validation, and test datasets."""
        total_samples = data.shape[0]  # Number of images
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)
        test_size = total_samples - train_size - val_size

        # Split along first dimension (num_images)
        train_data = data[:train_size]
        val_data = data[train_size:train_size+val_size]
        test_data = data[train_size+val_size:]

        print("\nDataset Information:")
        print(f"Total number of images: {total_samples}")
        print(f"Training: {train_data.shape[0]} images")
        print(f"Validation: {val_data.shape[0]} images")
        print(f"Test: {test_data.shape[0]} images")
        print(f"Image dimensions: {data.shape[1]}×{data.shape[2]} pixels")
        print(f"Wavelength points: {data.shape[3]}")

        # Create datasets
        print("\nCreating datasets...")
        train_dataset = FullImageHyperspectralDataset(train_data)
        val_dataset = FullImageHyperspectralDataset(val_data)
        test_dataset = FullImageHyperspectralDataset(test_data)

        # Visualize filter pattern and transmissions
        print("\nVisualizing filter arrangements...")
        train_dataset.visualize_filter_pattern(num_repeats=3)
        train_dataset.visualize_filter_transmissions()

        print("\nCreating data loaders...")
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        print(f"\nBatch size for training: {config.batch_size}")
        print(f"Number of training batches per epoch: {len(train_loader)}")

        return train_loader, val_loader, test_loader

    def visualize_reconstruction(self, model, test_loader, save_dir='results'):
        """Visualize reconstruction results."""
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        with torch.no_grad():
            # Get first test sample
            filtered_measurements, original_spectrum = next(iter(test_loader))
            filtered_measurements = filtered_measurements.to(self.device)
            original_spectrum = original_spectrum.to(self.device)

            # Generate reconstruction
            reconstructed_spectrum = model(filtered_measurements)

            # Move to CPU for plotting
            original_spectrum = original_spectrum.cpu().numpy()[0]
            reconstructed_spectrum = reconstructed_spectrum.cpu().numpy()[0]

            # Plot results
            wavelengths = np.linspace(1100, 1700, 9)

            # Plot central pixel spectrum
            h, w = original_spectrum.shape[1:]
            center_h, center_w = h//2, w//2

            plt.figure(figsize=(10, 6))
            plt.plot(wavelengths, original_spectrum[:, center_h, center_w], 'b-', label='Original')
            plt.plot(wavelengths, reconstructed_spectrum[:, center_h, center_w], 'r--', label='Reconstructed')
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            plt.title('Reconstruction Result (Center Pixel)')
            plt.legend()
            plt.grid(True)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plt.savefig(os.path.join(save_dir, f'reconstruction_{timestamp}.png'))
            plt.close()

            # Plot full image comparison
            wavelength_idx = 4  # Middle wavelength
            plt.figure(figsize=(15, 5))

            plt.subplot(131)
            plt.imshow(original_spectrum[wavelength_idx], cmap='viridis')
            plt.title('Original')
            plt.colorbar()

            plt.subplot(132)
            plt.imshow(reconstructed_spectrum[wavelength_idx], cmap='viridis')
            plt.title('Reconstructed')
            plt.colorbar()

            plt.subplot(133)
            difference = np.abs(original_spectrum[wavelength_idx] - reconstructed_spectrum[wavelength_idx])
            plt.imshow(difference, cmap='viridis')
            plt.title('Absolute Difference')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fullimage_comparison_{timestamp}.png'))
            plt.close()

def main(num_images=None):
    """Main training and evaluation pipeline."""
    print("Starting Hyperspectral Neural Network Training Pipeline...")
    print(f"\nConfiguration:")
    print(f"Number of filters: {config.num_filters}")
    print(f"Superpixel arrangement: {config.superpixel_height}×{config.superpixel_width}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    try:
        # Initialize processor
        processor = HyperspectralProcessor()

        # Load and prepare data
        print("\nLoading hyperspectral data...")
        all_data = processor.load_data(num_images)

        print("\nPreparing datasets...")
        train_loader, val_loader, test_loader = processor.prepare_datasets(all_data)

        # Initialize model
        print("\nInitializing model...")
        model = FullImageHyperspectralNet()
        print(f"Model architecture:\n{model}")

        # Initialize trainer
        trainer = Trainer(model, train_loader, val_loader)

        # Train model
        print("\nStarting training...")
        trainer.train()

        # Evaluate and visualize results
        print("\nEvaluating model...")
        test_loss, outputs, targets = trainer.evaluate_model(test_loader)
        print(f"Final test loss: {test_loss:.6f}")

        print("\nGenerating visualizations...")
        processor.visualize_reconstruction(model, test_loader)

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main(num_images=100)  # Process first 10 images
