import sys
import time
import numpy as np
import torch
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider, QLineEdit)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os
import rasterio

from config import config
from network import HyperspectralNet
from dataset import HyperspectralDataset

class ReconstructionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setWindowTitle('AVIRIS HSI Reconstruction Viewer')
        self.setGeometry(100, 100, 1600, 900)

        # Initialize variables
        self.full_reconstruction = None
        self.original_image = None
        self.wavelengths = None
        self.current_wavelength_idx = None

        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Create control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)

        # Add file selection
        self.file_combo = QComboBox()
        self.file_combo.addItems(self._get_available_files())
        control_layout.addWidget(QLabel("Select File:"))
        control_layout.addWidget(self.file_combo)

        # Add reconstruct button
        self.reconstruct_btn = QPushButton("Reconstruct")
        self.reconstruct_btn.clicked.connect(self.reconstruct_image)
        control_layout.addWidget(self.reconstruct_btn)

        # Add wavelength slider
        self.wavelength_slider = QSlider(Qt.Horizontal)
        self.wavelength_slider.setMinimum(0)
        self.wavelength_slider.setMaximum(len(config.wavelength_indices) - 1)
        self.wavelength_slider.setValue(len(config.wavelength_indices) // 2)
        self.wavelength_slider.valueChanged.connect(self.update_wavelength_display)
        control_layout.addWidget(QLabel("Wavelength:"))
        control_layout.addWidget(self.wavelength_slider)

        # Add wavelength display label
        self.wavelength_label = QLabel("Wavelength: N/A")
        control_layout.addWidget(self.wavelength_label)

        # Add coordinate inputs
        control_layout.addWidget(QLabel("X:"))
        self.x_coord = QLineEdit()
        self.x_coord.setFixedWidth(50)
        control_layout.addWidget(self.x_coord)

        control_layout.addWidget(QLabel("Y:"))
        self.y_coord = QLineEdit()
        self.y_coord.setFixedWidth(50)
        control_layout.addWidget(self.y_coord)

        self.plot_spectrum_btn = QPushButton("Plot Spectrum")
        self.plot_spectrum_btn.clicked.connect(self.plot_coordinates)
        control_layout.addWidget(self.plot_spectrum_btn)

        # Add RMSE display
        self.rmse_label = QLabel("RMSE: N/A")
        control_layout.addWidget(self.rmse_label)

        # Add time display
        self.time_label = QLabel("Time: N/A")
        control_layout.addWidget(self.time_label)

        layout.addWidget(control_panel)

        # Create matplotlib figures
        self.figure = plt.figure(figsize=(16, 8))
        self.gs = self.figure.add_gridspec(2, 2)
        self.ax_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_recon = self.figure.add_subplot(self.gs[0, 1])
        self.ax_spectrum = self.figure.add_subplot(self.gs[1, :])
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        # Connect click events
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Load the trained model
        checkpoint = torch.load(config.model_save_path)
        self.model = HyperspectralNet()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.filter_map = checkpoint['filter_map']
        self.model.eval()

    def _get_available_files(self):
        return sorted([f for f in os.listdir(config.dataset_path) if f.endswith('.tif')])

    def reconstruct_image(self):
        start_time = time.time()
        selected_file = self.file_combo.currentText()

        # Load hyperspectral data
        with rasterio.open(os.path.join(config.dataset_path, selected_file)) as src:
            data = src.read()
            data = np.transpose(data, (1, 2, 0))  # (H,W,C)

        # Store wavelengths and original image
        self.wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.original_image = data[:, :, config.wavelength_indices]
        self.original_image = self.original_image / np.max(self.original_image)

        # Create dataset with saved filter arrangement
        dataset = HyperspectralDataset(self.original_image)
        dataset.filter_map = self.filter_map
        filtered_measurements, _ = dataset[0]

        # Perform reconstruction
        with torch.no_grad():
            filtered_measurements = filtered_measurements.to(self.device)
            filtered_measurements = filtered_measurements.unsqueeze(0)
            reconstructed = self.model(filtered_measurements)
            self.full_reconstruction = reconstructed.squeeze(0).permute(1, 2, 0)

        # Calculate RMSE
        mse = torch.mean((self.full_reconstruction.cpu() - torch.tensor(self.original_image)) ** 2)
        rmse = torch.sqrt(mse)

        # Update time and RMSE labels
        elapsed_time = time.time() - start_time
        self.time_label.setText(f"Time: {elapsed_time:.2f}s")
        self.rmse_label.setText(f"RMSE: {rmse:.4f}")

        # Update display
        self.update_wavelength_display()

    def update_wavelength_display(self):
        if self.wavelengths is None:
            return

        idx = self.wavelength_slider.value()
        wavelength = self.wavelengths[idx]
        self.wavelength_label.setText(f"Wavelength: {wavelength:.2f} nm")
        self.display_images(idx)

    def display_images(self, idx):
        self.ax_orig.clear()
        self.ax_recon.clear()

        orig_img = self.original_image[:, :, idx]
        recon_img = self.full_reconstruction.cpu()[:, :, idx].numpy()

        self.ax_orig.imshow(orig_img, cmap='viridis')
        self.ax_recon.imshow(recon_img, cmap='viridis')

        self.ax_orig.set_title(f'Original Image ({self.wavelengths[idx]:.1f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({self.wavelengths[idx]:.1f}nm)')
        self.ax_orig.axis('off')
        self.ax_recon.axis('off')

        self.canvas.draw()

    def plot_coordinates(self):
        try:
            x = int(self.x_coord.text())
            y = int(self.y_coord.text())
            self.plot_spectrum(x, y)
        except ValueError:
            print("Invalid coordinates")

    def plot_spectrum(self, x, y):
        if not (0 <= y < self.original_image.shape[0] and
                0 <= x < self.original_image.shape[1]):
            return

        self.ax_spectrum.clear()
        self.ax_spectrum.plot(self.wavelengths,
                            self.original_image[y, x, :],
                            'b-', label='Original')
        self.ax_spectrum.plot(self.wavelengths,
                            self.full_reconstruction.cpu()[y, x, :].numpy(),
                            'r--', label='Reconstructed')

        self.ax_spectrum.set_xlabel('Wavelength (nm)')
        self.ax_spectrum.set_ylabel('Intensity')
        self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
        self.ax_spectrum.legend()
        self.ax_spectrum.grid(True)
        plt.ylim(0, 1)

        self.canvas.draw()

    def on_click(self, event):
        if event.inaxes in [self.ax_orig, self.ax_recon]:
            x, y = int(event.xdata), int(event.ydata)
            self.plot_spectrum(x, y)

def main():
    app = QApplication(sys.argv)
    viewer = ReconstructionViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()