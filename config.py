import os
import platform

class Config:
    def __init__(self):
        # Set base paths based on operating system
        if platform.system() == 'Windows':
            self.base_path = r"V:\SimulationData\Rahul\Hyperspectral Imaging Project"
        else:  # macOS
            self.base_path = '/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project'

        # Dataset paths
        self.dataset_path = os.path.join(self.base_path, 'HSI Data Sets',
                                       'HyperDrive_4wheelbuggyCapturedImages_SWIR')

        # Filter paths
        self.filter_path = os.path.join(self.base_path, 'Machine Learning Codes',
                                      'Filter CSV files', 'TransmissionTable_NIR.csv')

        # Model parameters
        self.batch_size = 1  # Changed to 1 since we're processing full images
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.num_filters = 6  # Number of different filters to use

        # Network architecture parameters
        self.conv_channels = [1, 64, 128, 256, 512]  # Conv channels
        self.num_wavelengths = 9
        self.input_channels = 1
        self.kernel_size = 3
        self.padding = 1
        self.use_batch_norm = True

        # Wavelength parameters
        self.swir_wavelengths = (1100, 1700, 9)  # start, end, points

        # Filter arrangement parameters
        self.num_arrangements = 2  # Number of random arrangements to try
        self.arrangement_seed = 42  # Base seed for reproducibility

        # Output paths
        self.model_save_path = 'models/Random_Filter_Arrangement_1100to1700_9filters.pth'
        self.results_path = 'results/021425'
        self.arrangements_path = os.path.join(self.results_path, 'arrangements')

config = Config()
