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
        # self.num_filters = 16
        self.num_filters = 64
        # self.superpixel_size = 4 # Change superpixel size here
        self.superpixel_size = 8 # Change superpixel size here
        self.conv_channels = [32, 64, 128]  # 3D Conv channels
        self.num_wavelengths = 19

        # Wavelength parameters
        self.vnir_wavelengths = (800, 900, 10)  # start, end, points
        self.swir_wavelengths = (1100, 1700, 9)  # start, end, points

        # Output paths
        self.model_save_path = 'models/012925_hyperspectral_model_800to1700.pth'
        self.results_path = 'results'

config = Config()
