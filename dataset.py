# Blah Blah

import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube):
        # First, let's pad the spatial dimensions if needed
        # Initialize class attributes first
        self.num_filters = config.num_filters
        self.superpixel_size = config.superpixel_size
        self.num_wavelengths = hyperspectral_cube.shape[-1]
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
        self.load_filter_data()

    def load_filter_data(self):
        # Read the CSV file, knowing first column is filter names
        self.filters = pd.read_csv(config.filter_path, header=None)

        # Calculate the wavelength spacing in the original CSV
        # 450 points span 800nm to 1700nm
        num_wavelength_points = 450
        csv_wavelengths = np.linspace(800, 1700, num_wavelength_points)
        wavelength_spacing = (1700 - 800) / (num_wavelength_points - 1)

        # Calculate indices for SWIR region (1100-1700nm)
        # Using the known spacing, we can calculate exact indices
        swir_start_idx = int((1100 - 800) / wavelength_spacing)  # Index for 1100nm
        swir_end_idx = num_wavelength_points  # Index for 1700nm

        # Extract filter transmissions for SWIR region
        # Add 1 to indices because first column is filter names
        filter_transmissions = self.filters.iloc[:self.num_filters,
                                               swir_start_idx+1:swir_end_idx+1].values

        # Get the actual wavelengths for our SWIR measurements
        cube_wavelengths = np.linspace(*config.swir_wavelengths)

        # Get the corresponding wavelengths from the CSV for the SWIR region
        csv_swir_wavelengths = csv_wavelengths[swir_start_idx:swir_end_idx]

        # Create our filter matrix through interpolation
        self.filter_matrix = self._interpolate_filters(
            filter_transmissions,
            csv_swir_wavelengths,  # Original wavelengths from CSV (1100-1700nm)
            cube_wavelengths       # Target wavelengths for our measurements
        ).float()

        # Add verification prints
        print(f"Original CSV wavelength range: 800nm to 1700nm ({num_wavelength_points} points)")
        print(f"SWIR wavelength range: {cube_wavelengths[0]:.1f}nm to {cube_wavelengths[-1]:.1f}nm")
        print(f"Number of SWIR wavelength points: {len(cube_wavelengths)}")
        print(f"Filter matrix shape: {self.filter_matrix.shape}")

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
