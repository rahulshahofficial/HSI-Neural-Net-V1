import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube, seed=42):
        """
        Args:
            hyperspectral_cube: Input data of shape (N,H,W,C) or (H,W,C)
            seed: Random seed for reproducible filter pattern
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_filters = config.num_filters

        if len(hyperspectral_cube.shape) == 3:
            self.num_images = 1
            hyperspectral_cube = hyperspectral_cube.reshape(1, *hyperspectral_cube.shape)
        else:
            self.num_images = hyperspectral_cube.shape[0]

        # Create random filter arrangement
        h, w = hyperspectral_cube.shape[1:3]
        self.filter_map = torch.randint(0, self.num_filters, (h, w))
        print(f"Created random filter arrangement of shape {self.filter_map.shape}")

        # Process images
        hyperspectral_cube = hyperspectral_cube / np.max(hyperspectral_cube)
        self.hypercube = torch.from_numpy(
            hyperspectral_cube[:, :, :, config.wavelength_indices]
        ).float()

        self.wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.load_filter_data()

        # Print filter distribution
        filter_counts = torch.bincount(self.filter_map.flatten(),
                                     minlength=self.num_filters)
        print("\nFilter distribution in random arrangement:")
        for i, count in enumerate(filter_counts):
            percentage = count.item() * 100 / (h * w)
            print(f"Filter {i}: {count.item()} pixels ({percentage:.1f}%)")

    def load_filter_data(self):
        filters_df = pd.read_csv(config.filter_path, header=None)

        if not hasattr(self, 'selected_filter_indices'):
            self.selected_filter_indices = np.random.choice(
                len(filters_df),
                size=self.num_filters,
                replace=False
            )

        filter_transmissions = filters_df.iloc[self.selected_filter_indices, 1:].values
        csv_wavelengths = np.linspace(800, 1700, filter_transmissions.shape[1])

        self.filter_matrix = self._interpolate_filters(
            filter_transmissions,
            csv_wavelengths,
            self.wavelengths
        ).float()

    def _interpolate_filters(self, filters, src_wavelengths, dst_wavelengths):
        interpolated = []
        for filter_spectrum in filters:
            interp = np.interp(dst_wavelengths, src_wavelengths, filter_spectrum)
            interpolated.append(interp)
        return torch.tensor(np.array(interpolated))

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        image = self.hypercube[idx]
        h, w, _ = image.shape

        # Create measurement tensor
        measurements = torch.zeros((h, w))

        # Calculate filtered measurements using random filter map
        for y in range(h):
            for x in range(w):
                filter_idx = self.filter_map[y, x]
                measurements[y, x] = torch.dot(
                    self.filter_matrix[filter_idx],
                    image[y, x, :]
                )

        filtered_measurements = measurements.unsqueeze(0)  # [1, H, W]
        spectral_cube = image.permute(2, 0, 1)  # [C, H, W]

        return filtered_measurements, spectral_cube

    def get_filter_arrangement(self):
        """Returns the random filter arrangement"""
        return self.filter_map.clone()

    def visualize_filter_pattern(self):
        """Visualizes the random filter arrangement"""
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 5))

        plt.subplot(121)
        sns.heatmap(self.filter_map, cmap='viridis',
                   cbar_kws={'label': 'Filter Index'})
        plt.title('Random Filter Arrangement')
        plt.xlabel('Width')
        plt.ylabel('Height')

        plt.subplot(122)
        filter_dist = torch.bincount(self.filter_map.flatten(),
                                   minlength=self.num_filters)
        plt.bar(range(len(filter_dist)), filter_dist)
        plt.title('Filter Distribution')
        plt.xlabel('Filter Index')
        plt.ylabel('Pixel Count')

        plt.tight_layout()
        plt.show()

    def visualize_filter_transmissions(self):
        """Visualizes the transmission spectra of selected filters"""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        for i in range(self.num_filters):
            plt.plot(self.wavelengths, self.filter_matrix[i],
                    label=f'Filter {i+1}', linewidth=2)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission')
        plt.title('Filter Transmission Spectra')
        plt.grid(True)
        plt.legend()
        plt.show()
