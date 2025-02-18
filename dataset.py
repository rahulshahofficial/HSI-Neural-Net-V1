import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config
import matplotlib.pyplot as plt


class FullImageHyperspectralDataset(Dataset):
    def __init__(self, hyperspectral_cube):
        """
        Initialize dataset with full hyperspectral images.
        Args:
            hyperspectral_cube: numpy array of shape (num_images, height, width, wavelengths)
                               or (height, width, wavelengths)
        """
        # Initialize class attributes
        self.num_filters = config.num_filters
        self.superpixel_height = config.superpixel_height
        self.superpixel_width = config.superpixel_width

        # Handle batch dimension
        if len(hyperspectral_cube.shape) == 4:
            self.num_images, h, w, wavelengths = hyperspectral_cube.shape
        else:
            h, w, wavelengths = hyperspectral_cube.shape
            self.num_images = 1
            hyperspectral_cube = hyperspectral_cube.reshape(1, h, w, wavelengths)

        self.num_wavelengths = wavelengths

        # Verify filter count doesn't exceed superpixel capacity
        max_filters = self.superpixel_height * self.superpixel_width
        if self.num_filters > max_filters:
            raise ValueError(f"Number of filters ({self.num_filters}) cannot exceed "
                           f"superpixel capacity ({max_filters})")

        # Calculate needed padding
        pad_h = (self.superpixel_height - (h % self.superpixel_height)) % self.superpixel_height
        pad_w = (self.superpixel_width - (w % self.superpixel_width)) % self.superpixel_width

        # Create padded cube if needed
        if pad_h > 0 or pad_w > 0:
            padded_cube = np.zeros((self.num_images, h + pad_h, w + pad_w, wavelengths))
            padded_cube[:, :h, :w, :] = hyperspectral_cube
            hyperspectral_cube = padded_cube
            print(f"Padded images from {h}×{w} to {h+pad_h}×{w+pad_w}")

        # Normalize and convert to tensor
        hyperspectral_cube = hyperspectral_cube / np.max(hyperspectral_cube)
        self.hypercube = torch.from_numpy(hyperspectral_cube).float()

        # Load and prepare filter data
        self.load_filter_data()

    def load_filter_data(self):
        """Load and process filter data from CSV."""
        # Read the CSV file
        self.filters = pd.read_csv(config.filter_path, header=None)

        # Calculate the wavelength spacing in the original CSV
        # 450 points span 800nm to 1700nm
        num_wavelength_points = 450
        csv_wavelengths = np.linspace(800, 1700, num_wavelength_points)
        wavelength_spacing = (1700 - 800) / (num_wavelength_points - 1)

        # Calculate indices for SWIR region (1100-1700nm)
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
            csv_swir_wavelengths,
            cube_wavelengths
        ).float()

        print(f"Loaded {self.num_filters} filters for wavelength range: "
              f"{cube_wavelengths[0]:.1f}nm to {cube_wavelengths[-1]:.1f}nm")

    def _interpolate_filters(self, filters, src_wavelengths, dst_wavelengths):
        """
        Interpolate filter transmissions to match measurement wavelengths.
        Args:
            filters: Filter transmission data
            src_wavelengths: Original wavelengths from CSV
            dst_wavelengths: Target wavelengths for measurements
        Returns:
            Interpolated filter matrix as tensor
        """
        interpolated = []
        for filter_spectrum in filters:
            interp = np.interp(dst_wavelengths, src_wavelengths, filter_spectrum)
            interpolated.append(interp)
        return torch.tensor(np.array(interpolated))

    def create_filtered_measurements(self, idx):
        """
        Create filtered measurements for a single image.
        Args:
            idx: Index of the image to process
        Returns:
            Tensor of filtered measurements
        """
        _, h, w, _ = self.hypercube.shape
        measurements = torch.zeros((1, h, w))  # 1 channel for intensity measurements

        # Apply filter pattern across the image
        for i in range(0, h, self.superpixel_height):
            for j in range(0, w, self.superpixel_width):
                for di in range(self.superpixel_height):
                    for dj in range(self.superpixel_width):
                        # Calculate filter index based on position within repeating pattern
                        filter_idx = (di * self.superpixel_width + dj) % self.num_filters

                        if filter_idx < self.num_filters:  # Ensure valid filter index
                            # Apply filter to current pixel
                            measurements[0, i+di, j+dj] = torch.dot(
                                self.filter_matrix[filter_idx],
                                self.hypercube[idx, i+di, j+dj, :]
                            )

        return measurements

    def __len__(self):
        """Return the number of images in the dataset."""
        return self.num_images

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        Args:
            idx: Index of the item to get
        Returns:
            Tuple of (filtered_measurements, original_spectrum)
        """
        filtered_measurements = self.create_filtered_measurements(idx)
        return filtered_measurements, self.hypercube[idx].permute(2, 0, 1)

    def visualize_filter_pattern(self, num_repeats=3):
        """
        Visualize the filter pattern and its repetition across the detector.
        Args:
            num_repeats: Number of times to show the pattern repeating in each direction
        """
        # Create base pattern
        base_pattern = np.zeros((self.superpixel_height, self.superpixel_width))
        for i in range(self.superpixel_height):
            for j in range(self.superpixel_width):
                filter_idx = (i * self.superpixel_width + j) % self.num_filters
                base_pattern[i, j] = filter_idx + 1  # +1 for better visualization

        # Create repeated pattern
        _, h, w, _ = self.hypercube.shape

        num_repeats_h = h // self.superpixel_height
        num_repeats_w = w // self.superpixel_width

        full_pattern = np.tile(base_pattern, (num_repeats_h, num_repeats_w))

        # Create the visualization
        plt.figure(figsize=(15, 8))

        # Plot single superpixel pattern
        plt.subplot(121)
        plt.imshow(base_pattern, cmap='viridis')
        plt.title('Single Superpixel Pattern (2x3)')
        plt.colorbar(label='Filter Index')

        # Add text annotations to show filter numbers
        for i in range(self.superpixel_height):
            for j in range(self.superpixel_width):
                plt.text(j, i, f'F{int(base_pattern[i,j])}',
                        ha='center', va='center', color='white')

        # Plot repeated pattern
        plt.subplot(122)
        plt.imshow(full_pattern, cmap='viridis')
        plt.title(f'Repeated Pattern ({num_repeats_h}x{num_repeats_w} superpixels)')
        plt.colorbar(label='Filter Index')

        # Add grid to show superpixel boundaries
        for i in range(num_repeats + 1):
            plt.axhline(y=i*self.superpixel_height - 0.5, color='w', linestyle='-', alpha=0.5)
            plt.axvline(x=i*self.superpixel_width - 0.5, color='w', linestyle='-', alpha=0.5)

        plt.tight_layout()
        plt.show()

    def visualize_filter_transmissions(self):
        """
        Visualize the transmission spectra of all filters.
        """
        wavelengths = np.linspace(*config.swir_wavelengths)

        plt.figure(figsize=(10, 6))
        for i in range(self.num_filters):
            plt.plot(wavelengths, self.filter_matrix[i],
                    label=f'Filter {i+1}', linewidth=2)

        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Transmission')
        plt.title('Filter Transmission Spectra')
        plt.grid(True)
        plt.legend()
        plt.show()
