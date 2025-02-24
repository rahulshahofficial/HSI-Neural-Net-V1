import os
import platform
import numpy as np


class Config:
    def __init__(self):
        # Set base paths based on operating system
        if platform.system() == 'Windows':
            self.base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        else:  # macOS
            self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        # Dataset paths
        self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'AVIRIS_augmented_dataset_2')

        # Filter paths
        self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
                                      'Filter CSV files', 'TransmissionTable_NIR.csv')

        # Model parameters
        self.batch_size = 64
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.num_filters = 16
        # self.num_filters = 16
        self.superpixel_height = 4
        self.superpixel_width = 4 # Change superpixel size here
        # self.superpixel_size = 4 # Change superpixel size here

        # Modified parameters for AVIRIS dataset
        self.image_height = 64  # For our cropped images
        self.image_width = 64
        self.num_wavelengths = 220  # AVIRIS has 220 bands
        self.wavelength_range = (800, 1700)  # nm, matching filter range
        # Generate full wavelength range
        self.full_wavelengths = np.linspace(400, 2500, 220)
        # Create mask for 800-1700nm range
        mask = (self.full_wavelengths >= 800) & (self.full_wavelengths <= 1700)
        self.wavelength_indices = np.where(mask)[0]
        self.num_output_wavelengths = len(self.wavelength_indices)
        self.input_channels = 1
        self.kernel_size = 3
        self.padding = 1
        self.use_batch_norm = True

        self.conv_channels = [1, 128, 256]  # 3D Conv channels

        # Output paths
        # self.num_filters = 64
        # self.superpixel_size = 8 # Change superpixel size here
        # self.model_save_path = 'models/013125_hyperspectral_model_800to1700.pth'

        self.num_filters = 16
        # self.superpixel_size = 3
        self.model_save_path = 'models/022425_AVIRIS_16filters_SRNet_64pix.pth'
        self.results_path = 'results/022425'

config = Config()
print(f"Number of wavelength indices: {len(config.wavelength_indices)}")
print(f"Range of indices: {min(config.wavelength_indices)} to {max(config.wavelength_indices)}")
print(f"Actual wavelengths: {config.full_wavelengths[config.wavelength_indices[0]]} to {config.full_wavelengths[config.wavelength_indices[-1]]}")
