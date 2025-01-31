'''@inproceedings{hanson2023hyperdrive,
  author={Hanson, Nathaniel and Pyatski, Benjamin and Hibbard, Samuel and DiMarzio, Charles and Padır, Taşkın},
  booktitle={2023 13th Workshop on Hyperspectral Imaging and Signal Processing: Evolution in Remote Sensing (WHISPERS)},
  title={Hyper-Drive: Visible-Short Wave Infrared Hyperspectral Imaging Datasets for Robots in Unstructured Environments},
  year={2023},
  volume={},
  number={},
  pages={1-5},
  keywords={Robot vision systems;Systems architecture;Ontologies;Signal processing;Cameras;Mobile robots;Hyperspectral imaging;hyperspectral imaging;robot spectroscopy;multimodal sensing;terrain segmentation},
  doi={10.1109/WHISPERS61460.2023.10430802}}
  '''
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from scipy.interpolate import interp1d
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QComboBox, QPushButton, QLabel,
                           QMessageBox, QFrame, QSpinBox, QCheckBox, QDoubleSpinBox)
from PyQt5.QtCore import Qt

class HyperspectralViewer:
    def __init__(self, vnir_path, swir_path, rgb_path, vnir_homography_path, swir_homography_path):
        """Initialize the hyperspectral data viewer."""
        # Load data
        self.vnir_cube = np.load(vnir_path).reshape((215, 407, 10))
        self.swir_cube = np.load(swir_path).reshape((168, 211, 9))
        self.rgb_img = cv2.imread(rgb_path)

        # Load homography matrices
        self.h_vnir = np.loadtxt(vnir_homography_path)
        self.h_swir = np.loadtxt(swir_homography_path)

        # Generate wavelength arrays
        self.vnir_wavelengths = np.linspace(800, 900, 10)
        self.swir_wavelengths = np.linspace(1100, 1700, 9)
        self.wavelengths = np.concatenate([self.vnir_wavelengths, self.swir_wavelengths])

        # Process the data
        self.combined_cube = None
        self.valid_mask = None
        self.bounds = None

    def apply_homography(self, img, homography, target_shape):
        """Apply homography transformation to an image."""
        return cv2.warpPerspective(
            img,
            homography,
            (target_shape[1], target_shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )

    def find_valid_region(self, mask):
        """Find the largest rectangular valid region in the mask."""
        rows, cols = mask.shape
        max_area = 0
        bounds = None

        heights = np.zeros(cols + 1, dtype=int)

        for row in range(rows):
            heights[:cols] = (heights[:cols] + 1) * mask[row, :]
            heights[cols] = 0

            stack = [-1]
            for col in range(cols + 1):
                while len(stack) > 1 and heights[col] < heights[stack[-1]]:
                    h = heights[stack.pop()]
                    w = col - stack[-1] - 1
                    area = h * w
                    if area > max_area:
                        max_area = area
                        bounds = (row - h + 1, stack[-1] + 1, row + 1, col)
                stack.append(col)

        return bounds

    def process_data(self):
        """Process and combine VNIR and SWIR data."""
        # Register VNIR data
        vnir_registered = np.stack([
            self.apply_homography(self.vnir_cube[:,:,i], self.h_vnir, self.rgb_img.shape[:2])
            for i in range(self.vnir_cube.shape[2])
        ], axis=2)

        # Register SWIR data
        swir_registered = np.stack([
            self.apply_homography(self.swir_cube[:,:,i], self.h_swir, self.rgb_img.shape[:2])
            for i in range(self.swir_cube.shape[2])
        ], axis=2)

        # Create valid masks
        vnir_mask = self.apply_homography(np.ones(self.vnir_cube.shape[:2]), self.h_vnir, self.rgb_img.shape[:2]) > 0.5
        swir_mask = self.apply_homography(np.ones(self.swir_cube.shape[:2]), self.h_swir, self.rgb_img.shape[:2]) > 0.5

        # Find valid region
        self.valid_mask = vnir_mask & swir_mask
        self.bounds = self.find_valid_region(self.valid_mask)

        if self.bounds is None:
            raise ValueError("No valid overlapping region found")

        # Combine and crop data
        r1, c1, r2, c2 = self.bounds
        self.combined_cube = np.dstack((
            vnir_registered[r1:r2, c1:c2],
            swir_registered[r1:r2, c1:c2]
        ))
        self.rgb_img = self.rgb_img[r1:r2, c1:c2]

    def get_wavelength_image(self, wavelength):
        """Get the image at the specified wavelength using interpolation."""
        if wavelength < self.wavelengths.min() or wavelength > self.wavelengths.max():
            return None

        # Find the closest wavelength indices
        wavelength_idx = np.abs(self.wavelengths - wavelength).argmin()

        # Return the image at that wavelength
        return self.combined_cube[:, :, wavelength_idx]

class MainWindow(QMainWindow):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.viewer = None
        self.max_intensity = None
        self.display_mode = 'rgb'  # Can be 'rgb' or 'wavelength'
        self.initUI()

    def _get_available_files(self):
        """Get list of files that exist in all three directories."""
        try:
            vnir_path = os.path.join(self.base_path, "VNIR_RAW")
            swir_path = os.path.join(self.base_path, "SWIR_RAW")
            rgb_path = os.path.join(self.base_path, "RGB_RAW")

            # Get list of files from each directory
            vnir_files = set(f.replace('.npy', '') for f in os.listdir(vnir_path) if f.endswith('.npy'))
            swir_files = set(f.replace('.npy', '') for f in os.listdir(swir_path) if f.endswith('.npy'))
            rgb_files = set(f.replace('.jpg', '') for f in os.listdir(rgb_path) if f.endswith('.jpg'))

            # Find common files
            common_files = list(vnir_files.intersection(swir_files, rgb_files))

            if not common_files:
                QMessageBox.warning(self, "Warning", "No matching files found in all three directories")
                return []

            return sorted(common_files)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading files: {str(e)}")
            return []

    def initUI(self):
        self.setWindowTitle('Hyperspectral Data Viewer')
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create control panel
        control_panel = QFrame()
        control_layout = QHBoxLayout(control_panel)

        # File selection
        self.file_combo = QComboBox()
        self.file_combo.addItems(self._get_available_files())
        control_layout.addWidget(QLabel("Select File:"))
        control_layout.addWidget(self.file_combo)

        # Load button
        load_button = QPushButton("Load Data")
        load_button.clicked.connect(self.load_data)
        control_layout.addWidget(load_button)

        # Wavelength selection
        wavelength_widget = QWidget()
        wavelength_layout = QHBoxLayout(wavelength_widget)
        wavelength_layout.addWidget(QLabel("Wavelength (nm):"))
        self.wavelength_spin = QDoubleSpinBox()
        self.wavelength_spin.setRange(800, 1700)
        self.wavelength_spin.setValue(800)
        self.wavelength_spin.setSingleStep(10)
        wavelength_layout.addWidget(self.wavelength_spin)

        # View wavelength button
        view_wavelength_button = QPushButton("View Wavelength")
        view_wavelength_button.clicked.connect(self.view_wavelength)
        wavelength_layout.addWidget(view_wavelength_button)

        # View RGB button
        view_rgb_button = QPushButton("View RGB")
        view_rgb_button.clicked.connect(self.view_rgb)
        wavelength_layout.addWidget(view_rgb_button)

        control_layout.addWidget(wavelength_widget)

        # Add coordinate input section
        coord_widget = QWidget()
        coord_layout = QHBoxLayout(coord_widget)

        coord_layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setMinimum(0)
        self.x_spin.setMaximum(9999)
        coord_layout.addWidget(self.x_spin)

        coord_layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setMinimum(0)
        self.y_spin.setMaximum(9999)
        coord_layout.addWidget(self.y_spin)

        view_coord_button = QPushButton("View Point")
        view_coord_button.clicked.connect(self.view_coordinates)
        coord_layout.addWidget(view_coord_button)

        control_layout.addWidget(coord_widget)
        layout.addWidget(control_panel)

        # Create matplotlib figure
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Initialize plots
        self.ax_img = self.figure.add_subplot(121)
        self.ax_spectrum = self.figure.add_subplot(122)
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Interpolation toggle
        interp_widget = QWidget()
        interp_layout = QHBoxLayout(interp_widget)
        self.interp_checkbox = QCheckBox("Interpolate Spectrum")
        self.interp_checkbox.setChecked(True)
        interp_layout.addWidget(self.interp_checkbox)
        control_layout.addWidget(interp_widget)

    def view_wavelength(self):
        """Display the image at the selected wavelength."""
        if not self.viewer:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        wavelength = self.wavelength_spin.value()
        wavelength_img = self.viewer.get_wavelength_image(wavelength)

        if wavelength_img is not None:
            self.display_mode = 'wavelength'
            self.ax_img.clear()
            im = self.ax_img.imshow(wavelength_img, cmap='viridis')
            self.ax_img.set_title(f'Image at {wavelength}nm')
            self.figure.colorbar(im, ax=self.ax_img)
            self.canvas.draw()
        else:
            QMessageBox.warning(self, "Error", "Invalid wavelength")

    def view_rgb(self):
        """Switch back to RGB view."""
        if not self.viewer:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        self.display_mode = 'rgb'
        self.ax_img.clear()
        self.ax_img.imshow(cv2.cvtColor(self.viewer.rgb_img, cv2.COLOR_BGR2RGB))
        self.ax_img.set_title('RGB Image')
        self.canvas.draw()

    def load_data(self):
        """Load and display the selected data."""
        try:
            selected_file = self.file_combo.currentText()
            if not selected_file:
                QMessageBox.warning(self, "Error", "Please select a file first")
                return

            # Create file paths
            vnir_file = os.path.join(self.base_path, "VNIR_RAW", f"{selected_file}.npy")
            swir_file = os.path.join(self.base_path, "SWIR_RAW", f"{selected_file}.npy")
            rgb_file = os.path.join(self.base_path, "RGB_RAW", f"{selected_file}.jpg")
            vnir_homography = os.path.join(self.base_path, "calibration", "ximea_to_rgb_homography.txt")
            swir_homography = os.path.join(self.base_path, "calibration", "imec_to_rgb_homography.txt")

            # Create and process viewer
            self.viewer = HyperspectralViewer(vnir_file, swir_file, rgb_file,
                                            vnir_homography, swir_homography)
            self.viewer.process_data()

            # Update spin box ranges
            self.x_spin.setMaximum(self.viewer.combined_cube.shape[1] - 1)
            self.y_spin.setMaximum(self.viewer.combined_cube.shape[0] - 1)

            # Find global maximum intensity
            self.max_intensity = np.max(self.viewer.combined_cube)

            # Display initial RGB image
            self.view_rgb()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")

    def view_coordinates(self):
        """Display spectrum for manually entered coordinates."""
        if not self.viewer:
            QMessageBox.warning(self, "Error", "Please load data first")
            return

        x = self.x_spin.value()
        y = self.y_spin.value()

        if 0 <= y < self.viewer.combined_cube.shape[0] and 0 <= x < self.viewer.combined_cube.shape[1]:
            self.plot_spectrum(x, y)
        else:
            QMessageBox.warning(self, "Error", "Coordinates out of range")


    def plot_spectrum(self, x, y):
        """Plot spectrum for given coordinates."""
        self.ax_spectrum.clear()
        spectrum = self.viewer.combined_cube[y, x]

        if self.interp_checkbox.isChecked():
            # Interpolated spectrum
            interp_wavelengths = np.linspace(
                self.viewer.wavelengths.min(),
                self.viewer.wavelengths.max(),
                1000
            )
            interp_spectrum = interp1d(
                self.viewer.wavelengths,
                spectrum,
                kind='cubic'
            )(interp_wavelengths)
            self.ax_spectrum.plot(interp_wavelengths, interp_spectrum, 'b-')
        else:
            # Original spectrum
            self.ax_spectrum.plot(self.viewer.wavelengths, spectrum, 'bo-')

        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title(f'Spectrum at pixel ({x}, {y})')
        self.ax_spectrum.grid(True)
        self.ax_spectrum.set_ylim([0, self.max_intensity])

        # Draw marker on image
        self.ax_img.clear()
        if self.display_mode == 'rgb':
            self.ax_img.imshow(cv2.cvtColor(self.viewer.rgb_img, cv2.COLOR_BGR2RGB))
            self.ax_img.set_title('RGB Image')
        else:  # wavelength mode
            wavelength = self.wavelength_spin.value()
            wavelength_img = self.viewer.get_wavelength_image(wavelength)
            im = self.ax_img.imshow(wavelength_img, cmap='viridis')
            self.ax_img.set_title(f'Image at {wavelength}nm')
            self.figure.colorbar(im, ax=self.ax_img)

        self.ax_img.plot(x, y, 'r+', markersize=10)
        self.canvas.draw()

    def on_click(self, event):
        """Handle click events on the image."""
        if not self.viewer or event.inaxes != self.ax_img:
            return

        x, y = int(event.xdata), int(event.ydata)
        if 0 <= y < self.viewer.combined_cube.shape[0] and 0 <= x < self.viewer.combined_cube.shape[1]:
            # Update spin boxes to match clicked coordinates
            self.x_spin.setValue(x)
            self.y_spin.setValue(y)
            # Plot the spectrum
            self.plot_spectrum(x, y)

def main():
    import sys
    base_path = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/HyperDrive_4wheelbuggyCapturedImages_SWIR"
    app = QApplication(sys.argv)
    main_window = MainWindow(base_path)
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
