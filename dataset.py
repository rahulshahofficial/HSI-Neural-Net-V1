import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config

class HyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_images, seed=42):
        """
        Args:
            hyperspectral_images: Either a single image (H,W,C) or multiple images (N,H,W,C)
            seed: Random seed for reproducible filter pattern
        """
        # Set random seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_filters = config.num_filters
        self.num_wavelengths = (hyperspectral_images.shape[-1]
                              if len(hyperspectral_images.shape) == 3
                              else hyperspectral_images.shape[-1])

        # Convert single image to list of images if necessary
        if len(hyperspectral_images.shape) == 3:
            hyperspectral_images = [hyperspectral_images]
        elif len(hyperspectral_images.shape) == 4:
            hyperspectral_images = [img for img in hyperspectral_images]

        # Process each image
        self.processed_images = []

        # Get dimensions from first image for filter map
        h, w, _ = hyperspectral_images[0].shape

        # Create single random filter map for all images
        self.filter_map = torch.randint(0, self.num_filters, (h, w))
        print(f"Created fixed random filter arrangement of shape {self.filter_map.shape}")

        for img in hyperspectral_images:
            # Verify image dimensions match filter map
            if img.shape[:2] != (h, w):
                raise ValueError(f"All images must have same dimensions. "
                               f"Expected {(h,w)}, got {img.shape[:2]}")
            # Normalize the image
            normalized_img = img / np.max(img)
            self.processed_images.append(torch.from_numpy(normalized_img).float())

        # Load filter data
        self.load_filter_data()

        print(f"Dataset initialized with {len(self.processed_images)} images")
        if len(self.processed_images) > 0:
            print(f"Image shape: {self.processed_images[0].shape}")

        # Print filter distribution statistics
        filter_counts = torch.bincount(self.filter_map.flatten(),
                                     minlength=self.num_filters)
        print("\nFilter distribution in random arrangement:")
        for i, count in enumerate(filter_counts):
            percentage = count.item() * 100 / (h * w)
            print(f"Filter {i}: {count.item()} pixels ({percentage:.1f}%)")

    def load_filter_data(self):
        # Read the CSV file
        filters_df = pd.read_csv(config.filter_path, header=None)

        # Calculate wavelength parameters
        num_wavelength_points = 450
        csv_wavelengths = np.linspace(800, 1700, num_wavelength_points)
        wavelength_spacing = (1700 - 800) / (num_wavelength_points - 1)

        # Calculate SWIR region indices
        swir_start_idx = int((1100 - 800) / wavelength_spacing)
        swir_end_idx = num_wavelength_points

        # If not already selected, randomly choose filters
        if not hasattr(self, 'selected_filter_indices'):
            self.selected_filter_indices = np.random.choice(
                len(filters_df),
                size=self.num_filters,
                replace=False
            )

        # Extract filter transmissions for SWIR region
        self.filters = filters_df.iloc[self.selected_filter_indices,
                                     swir_start_idx+1:swir_end_idx+1]

        # Get wavelengths for measurements
        cube_wavelengths = np.linspace(*config.swir_wavelengths)
        csv_swir_wavelengths = csv_wavelengths[swir_start_idx:swir_end_idx]

        # Create filter matrix
        filter_transmissions = self.filters.values
        self.filter_matrix = self._interpolate_filters(
            filter_transmissions,
            csv_swir_wavelengths,
            cube_wavelengths
        ).float()

        print(f"Filter matrix shape: {self.filter_matrix.shape}")

    def _interpolate_filters(self, filters, src_wavelengths, dst_wavelengths):
        interpolated = []
        for filter_spectrum in filters:
            interp = np.interp(dst_wavelengths, src_wavelengths, filter_spectrum)
            interpolated.append(interp)
        return torch.tensor(np.array(interpolated))

    def __len__(self):
        return len(self.processed_images)

    def __getitem__(self, idx):
        """Returns measurements and ground truth for a full image"""
        image = self.processed_images[idx]
        h, w, _ = image.shape

        # Create measurement tensor
        measurements = torch.zeros((h, w))

        # Calculate filtered measurements for each pixel using fixed filter map
        for y in range(h):
            for x in range(w):
                filter_idx = self.filter_map[y, x]
                measurements[y, x] = torch.dot(
                    self.filter_matrix[filter_idx],
                    image[y, x, :]
                )

        # Add channel dimension and return
        filtered_measurements = measurements.unsqueeze(0)  # [1, H, W]
        spectral_cube = image.permute(2, 0, 1)  # [C, H, W]

        return filtered_measurements, spectral_cube

    def get_filter_arrangement(self):
        """Returns the fixed random filter arrangement"""
        return self.filter_map.clone()
