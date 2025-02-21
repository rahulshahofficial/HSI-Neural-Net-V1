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
                                       'AVIRIS_augmented_dataset')

        # Filter paths
        self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
                                      'Filter CSV files', 'TransmissionTable_NIR.csv')

        # Model parameters
        self.batch_size = 1  # Changed to 1 since we're processing full images
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.num_filters = 16  # Number of different filters to use

        # Network architecture parameters
        self.conv_channels = [1, 64, 128, 256, 512]  # Conv channels
        self.kernel_size = 3
        self.padding = 1
        self.use_batch_norm = True

        # Modified parameters for AVIRIS dataset
        self.image_height = 100  # For our cropped images
        self.image_width = 100
        self.wavelength_range = (800, 1700)  # nm, matching filter range

        # Generate full wavelength range and indices
        self.full_wavelengths = np.linspace(400, 2500, 220)
        mask = (self.full_wavelengths >= 800) & (self.full_wavelengths <= 1700)
        self.wavelength_indices = np.where(mask)[0]
        self.num_wavelengths = len(self.wavelength_indices)

        # Random arrangement parameters
        self.num_arrangements =1 # Number of random arrangements to try
        self.arrangement_seed = 42  # Base seed for reproducibility

        # Output paths
        self.model_save_path = 'models/022124_AVIRIS_RandomArrangement.pth'
        self.results_path = 'results/022124'
        self.arrangements_path = os.path.join(self.results_path, 'arrangements')

config = Config()
