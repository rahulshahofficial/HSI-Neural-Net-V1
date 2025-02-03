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
    def __init__(self, vnir_path, swir_path):
        # First load and reshape to original dimensions
        vnir_data = np.load(vnir_path).reshape((215, 407, 24))

        # Select wavelengths between 800-900nm (indices 14-23, assuming linear spacing 660-900nm)
        start_idx = int(((800 - 660) / (900 - 660)) * 24)  # Calculate index for 800nm
        end_idx = start_idx + 10  # Take 10 points from there
        self.vnir_cube = vnir_data[:, :, start_idx:end_idx]

        # Load SWIR data
        self.swir_cube = np.load(swir_path).reshape((168, 211, 9))

        # Define wavelengths
        self.wavelengths = np.concatenate([
            np.linspace(800, 900, 10),
            np.linspace(1100, 1700, 9)
        ])

        self.combined_cube = self.combine_cubes()

    def combine_cubes(self):
        resized_swir = np.zeros((215, 407, 9))
        for i in range(9):
            resized_swir[:,:,i] = np.resize(self.swir_cube[:,:,i], (215, 407))
        return np.concatenate((self.vnir_cube, resized_swir), axis=2)

    @classmethod
    def get_all_files(cls):
        # base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/HyperDrive_4wheelbuggyCapturedImages_SWIR'
        base_path = "V:\SimulationData\Rahul\Hyperspectral Imaging Project\HSI Data Sets\HyperDrive_4wheelbuggyCapturedImages_SWIR"
        vnir_path = os.path.join(base_path, "VNIR_RAW")
        swir_path = os.path.join(base_path, "SWIR_RAW")
        vnir_files = {f.replace('.npy', '') for f in os.listdir(vnir_path) if f.endswith('.npy')}
        swir_files = {f.replace('.npy', '') for f in os.listdir(swir_path) if f.endswith('.npy')}
        common_files = vnir_files.intersection(swir_files)
        return {
            'files': sorted(list(common_files)),
            'vnir_path': vnir_path,
            'swir_path': swir_path
        }

    @classmethod
    def load_all_data(cls, num_images=None):
        paths = cls.get_all_files()
        combined_data = []
        for filename in paths['files'][:num_images]:
            print(f"Loading file: {filename}")
            vnir_file = os.path.join(paths['vnir_path'], f"{filename}.npy")
            swir_file = os.path.join(paths['swir_path'], f"{filename}.npy")
            viewer = HyperspectralViewer(vnir_file, swir_file)
            combined_data.append(viewer.combined_cube)
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
        wavelengths = np.concatenate([np.linspace(800, 900, 10), np.linspace(1100, 1700, 9)])
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
    main(num_images=250)
