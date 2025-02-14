import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import seaborn as sns
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QComboBox, QPushButton, QLabel,
                           QMessageBox, QFrame, QSpinBox)
from PyQt5.QtCore import Qt
import torch

from config import config
from dataset import HyperspectralDataset
from filter_arrangement import FilterArrangementEvaluator

class MainWindow(QMainWindow):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path
        self.viewer = None
        self.model = None
        self.filter_map = None
        self.original_data = None
        self.reconstructed_data = None
        self.wavelengths = np.linspace(1100, 1700, 9)

        self.initUI()

        # Load the best model if available
        self.load_best_model()

    def load_best_model(self):
        """Load the best model and filter arrangement."""
        try:
            model_path = 'results/best_model.pth'
            result = FilterArrangementEvaluator.load_best_model(model_path)
            self.model = result['model']
            self.filter_map = result['filter_map']
            print("Loaded best model and filter arrangement")
        except FileNotFoundError:
            print("No saved model found. Please train a model first.")

    def _get_available_files(self):
        """Get list of SWIR files."""
        try:
            swir_path = os.path.join(self.base_path, "SWIR_RAW")
            files = [f.replace('.npy', '') for f in os.listdir(swir_path)
                    if f.endswith('.npy')]
            return sorted(files)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error reading files: {str(e)}")
            return []

    def initUI(self):
        self.setWindowTitle('Hyperspectral Reconstruction Viewer')
        self.setGeometry(100, 100, 1600, 900)

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
        load_button = QPushButton("Load & Reconstruct")
        load_button.clicked.connect(self.load_and_reconstruct)
        control_layout.addWidget(load_button)

        # Wavelength selection
        control_layout.addWidget(QLabel("Wavelength Index:"))
        self.wavelength_spin = QSpinBox()
        self.wavelength_spin.setRange(0, 8)
        self.wavelength_spin.setValue(4)
        self.wavelength_spin.valueChanged.connect(self.update_wavelength_display)
        control_layout.addWidget(self.wavelength_spin)

        # Coordinate input
        control_layout.addWidget(QLabel("X:"))
        self.x_spin = QSpinBox()
        self.x_spin.setMinimum(0)
        self.x_spin.setMaximum(9999)
        control_layout.addWidget(self.x_spin)

        control_layout.addWidget(QLabel("Y:"))
        self.y_spin = QSpinBox()
        self.y_spin.setMinimum(0)
        self.y_spin.setMaximum(9999)
        control_layout.addWidget(self.y_spin)

        view_point_button = QPushButton("View Point")
        view_point_button.clicked.connect(self.view_coordinates)
        control_layout.addWidget(view_point_button)

        # Add error display
        self.error_label = QLabel("RMSE: N/A")
        control_layout.addWidget(self.error_label)

        layout.addWidget(control_panel)

        # Create matplotlib figure
        self.figure = Figure(figsize=(16, 8))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Add navigation toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        # Initialize plots
        gs = self.figure.add_gridspec(2, 2)
        self.ax_orig = self.figure.add_subplot(gs[0, 0])
        self.ax_recon = self.figure.add_subplot(gs[0, 1])
        self.ax_spectrum = self.figure.add_subplot(gs[1, :])

        # Connect click events
        self.canvas.mpl_connect('button_press_event', self.on_click)

    def load_and_reconstruct(self):
        """Load data and perform reconstruction."""
        if self.model is None:
            QMessageBox.warning(self, "Error", "No model loaded")
            return

        try:
            # Load SWIR data
            selected_file = self.file_combo.currentText()
            swir_path = os.path.join(self.base_path, "SWIR_RAW", f"{selected_file}.npy")
            swir_data = np.load(swir_path).reshape((168, 211, 9))

            # Create dataset with saved filter arrangement
            dataset = HyperspectralDataset(swir_data)

            # Get measurements and perform reconstruction
            with torch.no_grad():
                filtered_measurements, original_spectrum = dataset[0]
                reconstructed = self.model(filtered_measurements.unsqueeze(0)).squeeze()

            # Store data
            self.original_data = original_spectrum
            self.reconstructed_data = reconstructed

            # Calculate RMSE
            mse = torch.mean((reconstructed - original_spectrum) ** 2)
            rmse = torch.sqrt(mse)
            self.error_label.setText(f"RMSE: {rmse:.4f}")

            # Update coordinate ranges
            self.x_spin.setMaximum(self.original_data.shape[2] - 1)
            self.y_spin.setMaximum(self.original_data.shape[1] - 1)

            # Update display
            self.update_wavelength_display()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error loading data: {str(e)}")

    def update_wavelength_display(self):
        """Update image display for current wavelength."""
        if self.original_data is None or self.reconstructed_data is None:
            return

        wavelength_idx = self.wavelength_spin.value()
        wavelength = self.wavelengths[wavelength_idx]

        # Clear axes
        self.ax_orig.clear()
        self.ax_recon.clear()

        # Display original and reconstructed images
        orig_img = self.original_data[wavelength_idx].numpy()
        recon_img = self.reconstructed_data[wavelength_idx].numpy()

        im1 = self.ax_orig.imshow(orig_img, cmap='viridis')
        im2 = self.ax_recon.imshow(recon_img, cmap='viridis')

        self.ax_orig.set_title(f'Original Image ({wavelength:.0f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({wavelength:.0f}nm)')

        # Add colorbars
        self.figure.colorbar(im1, ax=self.ax_orig)
        self.figure.colorbar(im2, ax=self.ax_recon)

        self.canvas.draw()

    def view_coordinates(self):
        """Plot spectrum for manually entered coordinates."""
        if self.original_data is None:
            return

        x = self.x_spin.value()
        y = self.y_spin.value()

        if (0 <= x < self.original_data.shape[2] and
            0 <= y < self.original_data.shape[1]):
            self.plot_spectrum(x, y)
        else:
            QMessageBox.warning(self, "Error", "Coordinates out of range")

    def plot_spectrum(self, x, y):
        """Plot original and reconstructed spectra for given coordinates."""
        self.ax_spectrum.clear()

        # Get spectra
        orig_spectrum = self.original_data[:, y, x].numpy()
        recon_spectrum = self.reconstructed_data[:, y, x].numpy()

        # Plot spectra
        self.ax_spectrum.plot(self.wavelengths, orig_spectrum, 'b-', label='Original')
        self.ax_spectrum.plot(self.wavelengths, recon_spectrum, 'r--', label='Reconstructed')

        # Add labels and title
        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
        self.ax_spectrum.legend()
        self.ax_spectrum.grid(True)

        # Update markers on images
        self.ax_orig.plot(x, y, 'r+', markersize=10)
        self.ax_recon.plot(x, y, 'r+', markersize=10)

        self.canvas.draw()

    def on_click(self, event):
        """Handle click events to show spectrum at clicked point."""
        if event.inaxes in [self.ax_orig, self.ax_recon]:
            x, y = int(event.xdata), int(event.ydata)

            if (0 <= x < self.original_data.shape[2] and
                0 <= y < self.original_data.shape[1]):
                # Update spin boxes
                self.x_spin.setValue(x)
                self.y_spin.setValue(y)
                # Plot spectrum
                self.plot_spectrum(x, y)

def main():
    import sys
    base_path = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/HyperDrive_4wheelbuggyCapturedImages_SWIR"
    app = QApplication(sys.argv)
    viewer = MainWindow(base_path)
    viewer.show()
    sys.exit(app.exec_())
