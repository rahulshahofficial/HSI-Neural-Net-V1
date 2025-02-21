import os
import torch
import numpy as np
import rasterio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import config
from dataset import HyperspectralDataset
from network import HyperspectralNet
from filter_arrangement import FilterArrangementEvaluator

def load_data(num_images=None):
    """Load AVIRIS dataset"""
    data_dir = config.dataset_path
    files = sorted([f for f in os.listdir(data_dir) if f.endswith('.tif')])

    if num_images:
        files = files[:num_images]

    all_data = []
    print(f"Loading {len(files)} images...")

    for file in files:
        try:
            with rasterio.open(os.path.join(data_dir, file)) as src:
                data = src.read()  # Shape: (C,H,W)
                data = np.transpose(data, (1, 2, 0))  # Shape: (H,W,C)
                all_data.append(data)

            if len(all_data) % 50 == 0:
                print(f"Loaded {len(all_data)} images")

        except Exception as e:
            print(f"Error loading {file}: {str(e)}")
            continue

    return np.stack(all_data)

def main(num_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    print("Starting Hyperspectral Neural Network Training...")

    # Load AVIRIS data
    print("\nLoading dataset...")
    all_data = load_data(num_images)

    # Evaluate filter arrangements
    print(f"\nEvaluating {config.num_arrangements} different filter arrangements...")
    evaluator = FilterArrangementEvaluator(
        all_data,
        num_arrangements=config.num_arrangements
    )

    # Find best arrangement
    best_result = evaluator.evaluate_all()

    # Visualize arrangements
    evaluator.visualize_arrangements()
    evaluator.save_best_model()

    # Test best model
    print("\nTesting best model...")
    model = best_result['model']
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        dataset = HyperspectralDataset(all_data, seed=best_result['seed'])
        filtered_measurements, original_spectrum = dataset[0]
        filtered_measurements = filtered_measurements.to(device)  
        original_spectrum = original_spectrum.to(device)  # Add this
        reconstructed_spectrum = model(filtered_measurements.unsqueeze(0)).squeeze()

        # Center pixel spectrum
        h, w = original_spectrum.shape[1:]
        center_h, center_w = h//2, w//2
        orig_spectrum = original_spectrum[:, center_h, center_w].cpu().numpy()
        recon_spectrum = reconstructed_spectrum.cpu()[:, center_h, center_w].numpy()

        # Plot results
        os.makedirs(config.results_path, exist_ok=True)

        # Spectral comparison
        plt.figure(figsize=(10, 6))
        wavelengths = config.full_wavelengths[config.wavelength_indices]
        plt.plot(wavelengths, orig_spectrum, 'b-', label='Original')
        plt.plot(wavelengths, recon_spectrum, 'r--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Sample Reconstruction (Center Pixel)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(config.results_path, 'spectral_reconstruction.png'))
        plt.close()

        # Full image comparison
        mid_wavelength_idx = len(wavelengths) // 2
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(original_spectrum[mid_wavelength_idx].cpu().numpy())
        plt.title(f'Original ({wavelengths[mid_wavelength_idx]:.0f}nm)')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(reconstructed_spectrum.cpu()[mid_wavelength_idx].numpy())
        plt.title(f'Reconstructed ({wavelengths[mid_wavelength_idx]:.0f}nm)')
        plt.colorbar()

        plt.subplot(133)
        error = np.abs(original_spectrum[mid_wavelength_idx].cpu().numpy() -
                      reconstructed_spectrum.cpu()[mid_wavelength_idx].numpy())
        plt.imshow(error)
        plt.title('Absolute Error')
        plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(config.results_path, 'spatial_reconstruction.png'))
        plt.close()

        # Calculate error metrics
        mse = torch.mean((reconstructed_spectrum - original_spectrum) ** 2).item()
        print(f"\nMSE: {mse:.6f}")

if __name__ == "__main__":
    main()
