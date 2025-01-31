import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube):
        hyperspectral_cube = hyperspectral_cube / np.max(hyperspectral_cube)
        self.hypercube = torch.from_numpy(hyperspectral_cube).float()
        self.num_filters = config.num_filters
        self.superpixel_size = config.superpixel_size
        self.num_wavelengths = hyperspectral_cube.shape[-1]
        self.load_filter_data()

    def load_filter_data(self):
        self.filters = pd.read_csv(config.filter_path, header=None)
        filter_wavelengths = np.linspace(800, 1700, self.filters.shape[1]-1)
        filter_transmissions = self.filters.iloc[:self.num_filters, 1:].values

        cube_wavelengths = np.concatenate([
            np.linspace(800,900,10),
            np.linspace(*config.swir_wavelengths)
        ])
        self.filter_matrix = self._interpolate_filters(
            filter_transmissions,
            filter_wavelengths,
            cube_wavelengths
        ).float()

    def _interpolate_filters(self, filters, src_wavelengths, dst_wavelengths):
        interpolated = []
        for filter_spectrum in filters:
            interp = np.interp(dst_wavelengths, src_wavelengths, filter_spectrum)
            interpolated.append(interp)
        return torch.tensor(np.array(interpolated))

    def __len__(self):
        h, w, _ = self.hypercube.shape
        return (h // self.superpixel_size) * (w // self.superpixel_size)

    def __getitem__(self, idx):
        h, w, _ = self.hypercube.shape
        superpixels_per_row = w // self.superpixel_size
        y = (idx // superpixels_per_row) * self.superpixel_size
        x = (idx % superpixels_per_row) * self.superpixel_size

        # Get 8x8 superpixel
        superpixel = self.hypercube[y:y+self.superpixel_size, x:x+self.superpixel_size, :]

        # Create filtered measurements
        pixels = superpixel.reshape(-1, self.num_wavelengths)
        filtered_measurements = torch.matmul(self.filter_matrix, pixels.T)

        # Reshape filtered measurements: [channels, depth, height, width]
        filtered_measurements = filtered_measurements.reshape(
            self.num_filters,
            1,
            self.superpixel_size,
            self.superpixel_size
        )

        # Reshape target superpixel: [channels(wavelengths), height, width]
        superpixel = superpixel.permute(2, 0, 1)  # Move wavelengths to first dimension

        # filtered_measurements = filtered_measurements / filtered_measurements.max()
        # superpixel = superpixel / superpixel.max()
        return filtered_measurements, superpixel
