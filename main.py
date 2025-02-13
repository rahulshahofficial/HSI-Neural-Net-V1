import os
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np

from config import config
from dataset import HyperspectralDataset
from network import HyperspectralNet
from train import Trainer

class HyperspectralViewer:
    def __init__(self, swir_path):
        # Load SWIR data
        self.swir_cube = np.load(swir_path).reshape((168, 211, 9))

        # Define wavelengths
        self.wavelengths = np.linspace(1100, 1700, 9)

        # Pad the SWIR cube dimensions to be multiples of superpixel_size
        self.cube = self.pad_cube()

    def pad_cube(self):
        h, w, wavelengths = self.swir_cube.shape
        # Calculate padding needed to make dimensions multiples of superpixel_size
        pad_h = (config.superpixel_size - (h % config.superpixel_size)) % config.superpixel_size
        pad_w = (config.superpixel_size - (w % config.superpixel_size)) % config.superpixel_size

        # Create padded cube
        padded_cube = np.zeros((h + pad_h, w + pad_w, wavelengths))
        padded_cube[:h, :w, :] = self.swir_cube
        print(f"Padded SWIR cube from {h}×{w} to {h+pad_h}×{w+pad_w}")
        return padded_cube

    @classmethod
    def get_all_files(cls):
        # base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/HyperDrive_4wheelbuggyCapturedImages_SWIR'
        base_path = "V:\SimulationData\Rahul\Hyperspectral Imaging Project\HSI Data Sets\HyperDrive_4wheelbuggyCapturedImages_SWIR"
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
        return np.concatenate(combined_data, axis=0)

def main(num_images=None):
    torch.manual_seed(42)
    print("Starting Hyperspectral Neural Network Training...")

    print("Loading all hyperspectral data...")
    all_data = HyperspectralViewer.load_all_data(num_images)
    dataset = HyperspectralDataset(all_data)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    print(f"Created data loaders. Training samples: {train_size}, Validation samples: {val_size}")

    model = HyperspectralNet()
    print(f"Initialized model with {config.num_filters} filters and {all_data.shape[-1]} wavelength points")

    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()
    print("Training completed!")

    print("Testing reconstruction on sample point...")
    model.eval()
    with torch.no_grad():
        sample_idx = 1000
        filtered_measurements, original_spectrum = dataset[sample_idx]
        reconstructed_spectrum = model(filtered_measurements.unsqueeze(0)).squeeze()

        # Print shapes for debugging
        print(f"Original spectrum shape: {original_spectrum.shape}")
        print(f"Reconstructed spectrum shape: {reconstructed_spectrum.shape}")

        # Get the spectrum for the center pixel
        h, w = original_spectrum.shape[1:]  # Get spatial dimensions
        center_h, center_w = h//2, w//2     # Get center coordinates

        orig_spectrum = original_spectrum[:, center_h, center_w].numpy()
        recon_spectrum = reconstructed_spectrum[:, center_h, center_w].numpy()

        plt.figure(figsize=(10, 6))
        wavelengths = np.linspace(1100, 1700, 9)
        plt.plot(wavelengths, orig_spectrum, 'b-', label='Original')
        plt.plot(wavelengths, recon_spectrum, 'r--', label='Reconstructed')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Sample Reconstruction Result (Center Pixel)')
        plt.legend()
        plt.grid(True)

        os.makedirs(config.results_path, exist_ok=True)
        plt.savefig(os.path.join(config.results_path, 'sample_reconstruction.png'))
        plt.close()

    print(f"Results saved to {config.results_path}")

if __name__ == "__main__":
    main(num_images=100)
