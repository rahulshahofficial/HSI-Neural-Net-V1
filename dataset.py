import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from config import config
import matplotlib.pyplot as plt

class FullImageHyperspectralDataset(Dataset):
   def __init__(self, hyperspectral_cube):
       self.num_filters = config.num_filters
       self.superpixel_height = config.superpixel_height
       self.superpixel_width = config.superpixel_width

       if len(hyperspectral_cube.shape) == 4:
           self.num_images, h, w, wavelengths = hyperspectral_cube.shape
       else:
           h, w, wavelengths = hyperspectral_cube.shape
           self.num_images = 1
           hyperspectral_cube = hyperspectral_cube.reshape(1, h, w, wavelengths)

       max_filters = self.superpixel_height * self.superpixel_width
       if self.num_filters > max_filters:
           raise ValueError(f"Number of filters ({self.num_filters}) exceeds superpixel capacity ({max_filters})")

       pad_h = (self.superpixel_height - (h % self.superpixel_height)) % self.superpixel_height
       pad_w = (self.superpixel_width - (w % self.superpixel_width)) % self.superpixel_width

       if pad_h > 0 or pad_w > 0:
           padded_cube = np.zeros((self.num_images, h + pad_h, w + pad_w, wavelengths))
           padded_cube[:, :h, :w, :] = hyperspectral_cube
           hyperspectral_cube = padded_cube

       hyperspectral_cube = hyperspectral_cube / np.max(hyperspectral_cube)
       self.hypercube = torch.from_numpy(hyperspectral_cube[:, :, :, config.wavelength_indices]).float()
       self.wavelengths = config.full_wavelengths[config.wavelength_indices]
       self.num_wavelengths = len(self.wavelengths)
       self.load_filter_data()

   def load_filter_data(self):
       self.filters = pd.read_csv(config.filter_path, header=None)
       filter_transmissions = self.filters.iloc[:self.num_filters, 1:].values
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

   def create_filtered_measurements(self, idx):
       _, h, w, _ = self.hypercube.shape
       measurements = torch.zeros((1, h, w))

       for i in range(0, h, self.superpixel_height):
           for j in range(0, w, self.superpixel_width):
               for di in range(self.superpixel_height):
                   for dj in range(self.superpixel_width):
                       filter_idx = (di * self.superpixel_width + dj) % self.num_filters
                       if filter_idx < self.num_filters:
                           measurements[0, i+di, j+dj] = torch.dot(
                               self.filter_matrix[filter_idx],
                               self.hypercube[idx, i+di, j+dj, :]
                           )
       return measurements

   def __len__(self):
       return self.num_images

   def __getitem__(self, idx):
       filtered_measurements = self.create_filtered_measurements(idx)
       return filtered_measurements, self.hypercube[idx].permute(2, 0, 1)

   def visualize_filter_pattern(self, num_repeats=3):
       base_pattern = np.zeros((self.superpixel_height, self.superpixel_width))
       for i in range(self.superpixel_height):
           for j in range(self.superpixel_width):
               filter_idx = (i * self.superpixel_width + j) % self.num_filters
               base_pattern[i, j] = filter_idx + 1

       _, h, w, _ = self.hypercube.shape
       num_repeats_h = h // self.superpixel_height
       num_repeats_w = w // self.superpixel_width
       full_pattern = np.tile(base_pattern, (num_repeats_h, num_repeats_w))

       plt.figure(figsize=(15, 8))
       plt.subplot(121)
       plt.imshow(base_pattern, cmap='viridis')
       plt.title('Single Superpixel Pattern (2x3)')
       plt.colorbar(label='Filter Index')

       for i in range(self.superpixel_height):
           for j in range(self.superpixel_width):
               plt.text(j, i, f'F{int(base_pattern[i,j])}',
                       ha='center', va='center', color='white')

       plt.subplot(122)
       plt.imshow(full_pattern, cmap='viridis')
       plt.title(f'Repeated Pattern ({num_repeats_h}x{num_repeats_w} superpixels)')
       plt.colorbar(label='Filter Index')

       for i in range(num_repeats + 1):
           plt.axhline(y=i*self.superpixel_height - 0.5, color='w', linestyle='-', alpha=0.5)
           plt.axvline(x=i*self.superpixel_width - 0.5, color='w', linestyle='-', alpha=0.5)

       plt.tight_layout()
       plt.show()

   def visualize_filter_transmissions(self):
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
