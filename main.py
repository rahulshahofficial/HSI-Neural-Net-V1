import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import rasterio

from config import config
from dataset import FullImageHyperspectralDataset
from srnet_model import SpectralReconstructionNet
from train import Trainer

class HyperspectralProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def load_data(self, num_images=None):
        """Load pre-generated augmented dataset"""
        # Set data directory based on operating system
        if os.name == 'nt':  # Windows
            self.data_dir = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project\HSI Data Sets\AVIRIS_augmented_dataset"
        else:  # macOS or Linux
            self.data_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_augmented_dataset"

        # Make sure the directory exists, otherwise use a default location
        if not os.path.exists(self.data_dir):
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

        files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tif')])
        if num_images and num_images < len(files):
            files = files[:num_images]

        all_data = []
        print(f"Loading {len(files)} augmented images...")

        for file in files:
            try:
                with rasterio.open(os.path.join(self.data_dir, file)) as src:
                    # Read and transpose to (H,W,C) format
                    data = src.read()  # Shape: (C,H,W)
                    data = np.transpose(data, (1, 2, 0))  # Shape: (H,W,C)
                    all_data.append(data)

                if len(all_data) % 50 == 0:
                    print(f"Loaded {len(all_data)} images")

            except Exception as e:
                print(f"Error loading {file}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data could be loaded")

        return np.stack(all_data)


    def prepare_datasets(self, data):
        """Prepare training, validation, and test datasets."""
        total_samples = data.shape[0]
        train_size = int(0.8 * total_samples)
        val_size = int(0.1 * total_samples)

        # Split data
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

        # Visualize filter patterns
        print("\nVisualizing filter arrangements...")
        train_dataset.visualize_filter_pattern(num_repeats=3)
        train_dataset.visualize_filter_transmissions()

        # Build spectral dictionary if enabled
        try:
            from spect_dict import SpectralDictionary
            print("\nBuilding spectral dictionary from training data...")

            # Sample a subset of spectra from training data
            sample_indices = np.random.choice(
                train_data.shape[1] * train_data.shape[2],
                min(10000, train_data.shape[1] * train_data.shape[2]),
                replace=False
            )

            spectra_samples = []
            for i in range(min(train_data.shape[0], 50)):  # Use up to 50 images
                img_data = train_data[i]
                # Reshape to [H*W, C]
                img_spectra = img_data.reshape(-1, img_data.shape[2])
                # Sample spectra
                img_samples = img_spectra[sample_indices % img_spectra.shape[0]]
                spectra_samples.append(img_samples)

            # Combine all samples
            all_samples = np.vstack(spectra_samples)

            # Build dictionary
            spectral_dict = SpectralDictionary(n_components=20)
            spectral_dict.build_from_data(all_samples, force_rebuild=True)

        except Exception as e:
            print(f"Warning: Failed to build spectral dictionary: {str(e)}")

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        return train_loader, val_loader, test_loader

    def create_srnet_model(self):
        """Create and initialize the SRNet model"""
        # Get the number of output wavelengths from config
        num_wavelengths = config.num_output_wavelengths

        # Create SRNet model with appropriate configuration
        model = SpectralReconstructionNet(
            input_channels=1,  # Single channel of filtered measurements
            out_channels=num_wavelengths,  # Number of spectral bands to reconstruct
            dim=64,  # Base feature dimension (adjust based on available memory)
            deep_stage=3,  # Number of encoder/decoder stages
            num_blocks=[1, 2, 3],  # Increasing number of SAM blocks with depth
            num_heads=[2, 4, 8]  # Increasing number of attention heads with depth
        )

        print(f"SRNet Configuration:")
        print(f"  Base dimension: 64")
        print(f"  Number of stages: 3")
        print(f"  Attention blocks per stage: [1, 2, 3]")
        print(f"  Attention heads per stage: [2, 4, 8]")
        print(f"  Output wavelengths: {num_wavelengths}")

        return model

    def visualize_reconstruction(self, model, test_loader, save_dir='results'):
        """Visualize reconstruction results."""
        os.makedirs(save_dir, exist_ok=True)
        model.eval()

        with torch.no_grad():
            # Get a test sample
            filtered_measurements, filter_pattern, original_spectrum = next(iter(test_loader))

            # Move to device
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)
            original_spectrum = original_spectrum.to(self.device)

            # Perform reconstruction
            reconstructed_spectrum = model(filtered_measurements, filter_pattern)

            # Move to CPU for plotting
            original_spectrum = original_spectrum.cpu().numpy()[0]
            reconstructed_spectrum = reconstructed_spectrum.cpu().numpy()[0]

            # Plot results for wavelengths in range (800-1700nm)
            wavelengths = config.full_wavelengths[config.wavelength_indices]

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

            # Plot full image comparison at middle wavelength
            middle_idx = len(wavelengths) // 2
            middle_wavelength = wavelengths[middle_idx]

            plt.figure(figsize=(15, 5))
            plt.subplot(131)
            plt.imshow(original_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Original at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(132)
            plt.imshow(reconstructed_spectrum[middle_idx], cmap='viridis')
            plt.title(f'Reconstructed at {middle_wavelength:.0f}nm')
            plt.colorbar()

            plt.subplot(133)
            difference = np.abs(original_spectrum[middle_idx] - reconstructed_spectrum[middle_idx])
            plt.imshow(difference, cmap='viridis')
            plt.title('Absolute Difference')
            plt.colorbar()

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'fullimage_comparison_{timestamp}.png'))
            plt.close()


def main():
    """Main training and evaluation pipeline."""
    num_images = 250

    print("Starting Hyperspectral Neural Network Training Pipeline with SRNet...")
    print(f"\nConfiguration:")
    print(f"Number of filters: {config.num_filters}")
    print(f"Superpixel arrangement: {config.superpixel_height}×{config.superpixel_width}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Number of epochs: {config.num_epochs}")
    print(f"Batch size: {config.batch_size}")

    try:
        processor = HyperspectralProcessor()

        print("\nLoading augmented dataset...")
        all_data = processor.load_data(num_images)

        print("\nPreparing datasets...")
        train_loader, val_loader, test_loader = processor.prepare_datasets(all_data)

        print("\nInitializing SRNet model...")
        model = processor.create_srnet_model()
        print(f"Model created successfully")

        trainer = Trainer(model, train_loader, val_loader)

        print("\nStarting training...")
        trainer.train()

        print("\nEvaluating model...")
        test_loss, outputs, targets = trainer.evaluate_model(test_loader)
        print(f"Final test loss: {test_loss['total_loss']:.6f}")


        print("\nGenerating visualizations...")
        processor.visualize_reconstruction(model, test_loader)

        print("\nPipeline completed successfully!")

    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
