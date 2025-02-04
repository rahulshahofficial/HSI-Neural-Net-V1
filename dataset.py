# Blah Blah

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube):
        # First, let's pad the spatial dimensions if needed
        h, w, wavelengths = hyperspectral_cube.shape

        # Calculate needed padding
        pad_h = (self.superpixel_size - (h % self.superpixel_size)) % self.superpixel_size
        pad_w = (self.superpixel_size - (w % self.superpixel_size)) % self.superpixel_size

        # Create padded cube if needed
        if pad_h > 0 or pad_w > 0:
            padded_cube = np.zeros((h + pad_h, w + pad_w, wavelengths))
            padded_cube[:h, :w, :] = hyperspectral_cube
            hyperspectral_cube = padded_cube
            print(f"Padded image from {h}×{w} to {h+pad_h}×{w+pad_w} to ensure complete filter patterns")

        # Normalize and convert to tensor
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

        superpixel = self.hypercube[y:y+self.superpixel_size, x:x+self.superpixel_size, :]
        measurements = torch.zeros((self.superpixel_size, self.superpixel_size))

        # Calculate base position within the repeating pattern
        pattern_y = y % self.superpixel_size
        pattern_x = x % self.superpixel_size

        for i in range(self.superpixel_size):
            for j in range(self.superpixel_size):
                # Calculate filter index based on position within repeating pattern
                filter_idx = ((i + pattern_y) % self.superpixel_size) * self.superpixel_size + \
                            ((j + pattern_x) % self.superpixel_size)

                measurements[i,j] = torch.dot(
                    self.filter_matrix[filter_idx],
                    superpixel[i,j,:]
                )

        filtered_measurements = measurements.unsqueeze(0)
        superpixel = superpixel.permute(2, 0, 1)
        return filtered_measurements, superpixel
