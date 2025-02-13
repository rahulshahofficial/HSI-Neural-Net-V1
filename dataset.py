import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube, seed=42):
        self.num_filters = config.num_filters
        self.num_wavelengths = hyperspectral_cube.shape[-1]

        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Normalize the cube
        self.hypercube = torch.from_numpy(
            hyperspectral_cube / np.max(hyperspectral_cube)
        ).float()

        # Create random filter assignment for each pixel
        h, w, _ = hyperspectral_cube.shape
        self.random_filter_map = torch.randint(
            0, self.num_filters, (h, w)
        )

        self.load_filter_data()

        # Store dimensions for later use
        self.height = h
        self.width = w

    def load_filter_data(self):
        # Read the CSV file
        self.filters = pd.read_csv(config.filter_path, header=None)

        # Calculate wavelength parameters
        num_wavelength_points = 450
        csv_wavelengths = np.linspace(800, 1700, num_wavelength_points)
        wavelength_spacing = (1700 - 800) / (num_wavelength_points - 1)

        # Calculate SWIR region indices
        swir_start_idx = int((1100 - 800) / wavelength_spacing)
        swir_end_idx = num_wavelength_points

        # Extract and interpolate filter transmissions
        filter_transmissions = self.filters.iloc[:self.num_filters,
                                               swir_start_idx+1:swir_end_idx+1].values

        # Get wavelengths
        cube_wavelengths = np.linspace(*config.swir_wavelengths)
        csv_swir_wavelengths = csv_wavelengths[swir_start_idx:swir_end_idx]

        # Create filter matrix
        self.filter_matrix = self._interpolate_filters(
            filter_transmissions,
            csv_swir_wavelengths,
            cube_wavelengths
        ).float()

    def _interpolate_filters(self, filters, src_wavelengths, dst_wavelengths):
        interpolated = []
        for filter_spectrum in filters:
            interp = np.interp(dst_wavelengths, src_wavelengths, filter_spectrum)
            interpolated.append(interp)
        return torch.tensor(np.array(interpolated))

    def __len__(self):
        return 1  # Each item is a full image

    def __getitem__(self, idx):
        # Create measurement tensor
        measurements = torch.zeros((self.height, self.width))

        # Calculate filtered measurements for each pixel
        for y in range(self.height):
            for x in range(self.width):
                filter_idx = self.random_filter_map[y, x]
                measurements[y, x] = torch.dot(
                    self.filter_matrix[filter_idx],
                    self.hypercube[y, x, :]
                )

        # Add channel dimension and return
        filtered_measurements = measurements.unsqueeze(0)  # [1, H, W]
        spectral_cube = self.hypercube.permute(2, 0, 1)  # [W, H, C] -> [C, H, W]

        return filtered_measurements, spectral_cube
