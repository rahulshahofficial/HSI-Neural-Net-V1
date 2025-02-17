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

        # Calibration paths
        self.vnir_homography = os.path.join(self.dataset_path, 'calibration',
                                          'ximea_to_rgb_homography.txt')
        self.swir_homography = os.path.join(self.dataset_path, 'calibration',
                                          'imec_to_rgb_homography.txt')
        self.rgb_path = os.path.join(self.dataset_path, "RGB_RAW")

        # Model parameters
        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 1e-4
        self.num_filters = 6
        # self.num_filters = 16
        self.superpixel_height = 2
        self.superpixel_width = 3  # Change superpixel size here
        # self.superpixel_size = 4 # Change superpixel size here
        self.conv_channels = [1, 128, 256]  # 3D Conv channels
        self.num_wavelengths = 9
        self.input_channels = 1  # New parameter to make it clear we have 1 input channel
        self.kernel_size = 3           # Make convolution kernel size configurable
        self.padding = 1              # Make padding configurable
        self.use_batch_norm = True    # Control batch normalization

        # Wavelength parameters
        self.swir_wavelengths = (1100, 1700, 9)  # start, end, points

        # Output paths
        # self.num_filters = 64
        # self.superpixel_size = 8 # Change superpixel size here
        # self.model_save_path = 'models/013125_hyperspectral_model_800to1700.pth'

        self.num_filters = 6
        # self.superpixel_size = 3
        self.model_save_path = 'models/021725_hyperspectral_model_9filters_FULLIMAGE.pth'
        self.results_path = 'results/021725'

config = Config()
