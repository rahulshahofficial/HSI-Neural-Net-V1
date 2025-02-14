import os
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.function as F
import matplotlib.pyplot as plt
import numpy as np

from config import config
from dataset import HyperspectralDataset
from network import HyperspectralNet
from train import Trainer
from filter_arrangement import FilterArrangementEvaluator

class HyperspectralViewer:
    def __init__(self, swir_path):
        # Load SWIR data
        self.swir_cube = np.load(swir_path).reshape((168, 211, 9))
        self.wavelengths = np.linspace(1100, 1700, 9)
        # Normalize the cube
        self.cube = self.swir_cube / np.max(self.swir_cube)

    @classmethod
    def get_all_files(cls):
        base_path = config.dataset_path
        swir_path = os.path.join(base_path, "SWIR_RAW")
        return {
            'files': sorted([f.replace('.npy', '') for f in os.listdir(swir_path)
                           if f.endswith('.npy')]),
            'swir_path': swir_path
        }

    @classmethod
    def load_all_data(cls, num_images=None):
        paths = cls.get_all_files()
        combined_data = []
        for filename in paths['files'][:num_images]:
            print(f"Loading file: {filename}")
            swir_file = os.path.join(paths['swir_path'], f"{filename}.npy")
            viewer = HyperspectralViewer(swir_file)
            combined_data.append(viewer.cube)
        return np.stack(combined_data)  # Stack preserves image structure

def main(num_images=100, num_arrangements=5):
    torch.manual_seed(42)
    print("Starting Hyperspectral Neural Network Training...")

    # Load data
    print("Loading all hyperspectral data...")
    all_data = HyperspectralViewer.load_all_data(num_images)
    if len(all_data.shape) == 3:
        all_data = all_data.reshape(1, *all_data.shape)

    # Create evaluator
    print(f"\nEvaluating {num_arrangements} different filter arrangements...")
    evaluator = FilterArrangementEvaluator(
        all_data,
        HyperspectralNet,
        num_arrangements=num_arrangements
    )

    # Find best arrangement and get trained model
    best_result = evaluator.find_best_arrangement()
    best_model = evaluator.best_model

    # Visualize filter arrangements
    evaluator.visualize_arrangements()

    # Save best model and arrangement
    evaluator.save_best_model()

    print("\nTesting best model on validation data...")
    best_model.eval()
    with torch.no_grad():
        # Create dataset with best arrangement
        dataset = HyperspectralDataset(all_data, seed=best_result['seed'])

        # Get a sample image
        filtered_measurements, original_spectrum = dataset[0]
        reconstructed_spectrum = best_model(filtered_measurements.unsqueeze(0)).squeeze()

        # Print shapes for debugging
        print(f"\nDebug Information:")
        print(f"Original spectrum shape: {original_spectrum.shape}")
        print(f"Reconstructed spectrum shape: {reconstructed_spectrum.shape}")

        # Visualize center pixel spectrum
        h, w = original_spectrum.shape[1:]
        center_h, center_w = h//2, w//2

        # Extract center pixel spectra
        orig_spectrum = original_spectrum[:, center_h, center_w].numpy()
        recon_spectrum = reconstructed_spectrum[:, center_h, center_w].numpy()

        # Plot spectral comparison
        plt.figure(figsize=(10, 6))
        wavelengths = np.linspace(1100, 1700, 9)
        plt.plot(wavelengths, orig_spectrum, 'b-', label='Original')
        plt.plot(wavelengths, recon_spectrum, 'r--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Sample Reconstruction Result (Center Pixel)')
        plt.legend()
        plt.grid(True)

        # Save spectral comparison
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/sample_reconstruction.png')
        plt.close()

        # Visualize full image reconstruction at middle wavelength
        mid_wavelength_idx = original_spectrum.shape[0] // 2

        plt.figure(figsize=(15, 5))

        # Original image
        plt.subplot(131)
        plt.imshow(original_spectrum[mid_wavelength_idx].numpy())
        plt.title(f'Original (λ={wavelengths[mid_wavelength_idx]:.0f}nm)')
        plt.colorbar()

        # Reconstructed image
        plt.subplot(132)
        plt.imshow(reconstructed_spectrum[mid_wavelength_idx].numpy())
        plt.title(f'Reconstructed (λ={wavelengths[mid_wavelength_idx]:.0f}nm)')
        plt.colorbar()

        # Error map
        error = np.abs(original_spectrum[mid_wavelength_idx].numpy() -
                      reconstructed_spectrum[mid_wavelength_idx].numpy())
        plt.subplot(133)
        plt.imshow(error)
        plt.title('Absolute Error')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig('results/full_image_comparison.png')
        plt.close()

        # Calculate and print error metrics
        mse = F.mse_loss(reconstructed_spectrum, original_spectrum).item()
        mae = F.l1_loss(reconstructed_spectrum, original_spectrum).item()

        print("\nError Metrics:")
        print(f"MSE: {mse:.6f}")
        print(f"MAE: {mae:.6f}")

    print("\nResults saved in 'results' directory")

if __name__ == "__main__":
    # Can adjust these parameters
    main(num_images=100, num_arrangements=5)
