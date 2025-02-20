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
from network import FullImageHyperspectralNet
from dataset import FullImageHyperspectralDataset

class ReconstructionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AVIRIS HSI Reconstruction Viewer')
        self.setGeometry(100, 100, 1600, 900)

        # Initialize variables
        self.full_reconstruction = None
        self.original_image = None
        self.wavelengths = None
        self.current_wavelength_idx = None
        self.data_dir = "/Volumes/ValentineLab-1/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_augmented_dataset"

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
        self.wavelength_slider.setMinimum(800)
        self.wavelength_slider.setMaximum(1700)
        self.wavelength_slider.setValue(1250)
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

        # Add plot spectrum button
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
        self.model = FullImageHyperspectralNet()
        self.model.load_state_dict(torch.load(config.model_save_path, map_location='cpu'))
        self.model.eval()

    def _get_available_files(self):
        """Get list of files from augmented dataset directory."""
        return sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tif')])

    def reconstruct_image(self):
        """Reconstruct the selected image."""
        start_time = time.time()
        selected_file = self.file_combo.currentText()

        # Load hyperspectral data
        with rasterio.open(os.path.join(self.data_dir, selected_file)) as src:
            img_data = src.read()
            img_data = np.transpose(img_data, (1, 2, 0))  # Change to (H,W,C)
            img_data = img_data.reshape((1, *img_data.shape))  # Add batch dimension

        # Create dataset with single image
        dataset = FullImageHyperspectralDataset(img_data)
        filtered_measurements, original_spectrum = dataset[0]

        # Store wavelengths and original image
        self.wavelengths = config.full_wavelengths[config.wavelength_indices]
        self.original_image = original_spectrum.numpy()

        # Perform reconstruction
        with torch.no_grad():
            filtered_measurements = filtered_measurements.unsqueeze(0)
            reconstructed = self.model(filtered_measurements)
            self.full_reconstruction = reconstructed.squeeze(0)

        # Calculate RMSE
        mse = torch.mean((self.full_reconstruction - torch.tensor(self.original_image)) ** 2)
        rmse = torch.sqrt(mse)

        # Update time and RMSE labels
        elapsed_time = time.time() - start_time
        self.time_label.setText(f"Time: {elapsed_time:.2f}s")
        self.rmse_label.setText(f"RMSE: {rmse:.4f}")

        # Update display
        self.update_wavelength_display()

    def update_wavelength_display(self):
        """Update display when wavelength slider changes."""
        if self.wavelengths is None or self.original_image is None:
            return

        idx = self.wavelength_slider.value()
        wavelength = self.wavelengths[idx]
        self.wavelength_label.setText(f"Wavelength: {wavelength:.2f} nm")

        # Update images
        self.ax_orig.clear()
        self.ax_recon.clear()

        orig_img = self.original_image[idx]
        recon_img = self.full_reconstruction[idx].numpy()

        self.ax_orig.imshow(orig_img, cmap='viridis')
        self.ax_recon.imshow(recon_img, cmap='viridis')

        self.ax_orig.set_title(f'Original Image ({wavelength:.1f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({wavelength:.1f}nm)')
        self.ax_orig.axis('off')
        self.ax_recon.axis('off')

        self.canvas.draw()

    def plot_coordinates(self):
        """Plot spectrum for manually entered coordinates."""
        try:
            x = int(self.x_coord.text())
            y = int(self.y_coord.text())
            self.plot_spectrum(x, y)
        except ValueError:
            print("Invalid coordinates")

    def plot_spectrum(self, x, y):
        """Plot spectrum for given coordinates."""
        if 0 <= y < self.original_image.shape[1] and 0 <= x < self.original_image.shape[2]:
            self.ax_spectrum.clear()

            # Plot original and reconstructed spectra
            self.ax_spectrum.plot(self.wavelengths,
                                self.original_image[:, y, x],
                                'b-', label='Original')
            self.ax_spectrum.plot(self.wavelengths,
                                self.full_reconstruction[:, y, x].numpy(),
                                'r--', label='Reconstructed')

            self.ax_spectrum.set_xlabel('Wavelength (nm)')
            self.ax_spectrum.set_ylabel('Intensity')
            self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
            self.ax_spectrum.legend()
            self.ax_spectrum.grid(True)
            plt.ylim(0, 1)

            self.canvas.draw()

    def on_click(self, event):
        """Handle click events to show spectrum at clicked point."""
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
