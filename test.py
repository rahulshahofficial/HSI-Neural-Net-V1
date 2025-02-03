from config import config
from network import HyperspectralNet
from dataset import HyperspectralDataset
from main import HyperspectralViewer
from viewer import HyperspectralViewer

import sys
import time
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider, QLineEdit)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import torch
from torch.utils.data import DataLoader
import os
import numpy as np

class ReconstructionViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Hyperspectral Reconstruction Viewer')
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
        self.wavelength_slider.setMaximum(18)  # 19 wavelengths - 1
        self.wavelength_slider.setValue(9)  # Middle wavelength
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
        self.model = HyperspectralNet()
        self.model.load_state_dict(torch.load(config.model_save_path, weights_only=True))
        self.model.eval()

    def plot_coordinates(self):
        """Plot spectrum for manually entered coordinates."""
        try:
            x = int(self.x_coord.text())
            y = int(self.y_coord.text())

            if (0 <= x < self.original_image.shape[1] and
                0 <= y < self.original_image.shape[0]):
                self.ax_spectrum.clear()

                self.ax_spectrum.plot(self.wavelengths,
                                    self.original_image[y,x,:],
                                    'b-', label='Original')
                self.ax_spectrum.plot(self.wavelengths,
                                    self.full_reconstruction[y,x,:].numpy(),
                                    'r--', label='Reconstructed')

                self.ax_spectrum.set_xlabel('Wavelength (nm)')
                self.ax_spectrum.set_ylabel('Intensity')
                self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
                self.ax_spectrum.legend()
                self.ax_spectrum.grid(True)
                plt.ylim(0,1)

                self.canvas.draw()
            else:
                print("Coordinates out of range")
        except ValueError:
            print("Invalid coordinates")

    def _get_available_files(self):
        """Get list of files common to both VNIR and SWIR directories."""
        vnir_path = os.path.join(config.dataset_path, "VNIR_RAW")
        swir_path = os.path.join(config.dataset_path, "SWIR_RAW")

        vnir_files = {f.replace('.npy', '') for f in os.listdir(vnir_path)
                     if f.endswith('.npy')}
        swir_files = {f.replace('.npy', '') for f in os.listdir(swir_path)
                     if f.endswith('.npy')}

        return sorted(list(vnir_files.intersection(swir_files)))

    def update_wavelength_display(self):
        """Update display when wavelength slider changes."""
        if self.wavelengths is not None:
            wavelength = self.wavelengths[self.wavelength_slider.value()]
            self.wavelength_label.setText(f"Wavelength: {wavelength:.2f} nm")
            self.update_images()

    def update_images(self):
        """Update image display for current wavelength."""
        if self.original_image is None or self.full_reconstruction is None:
            return

        idx = self.wavelength_slider.value()
        wavelength = self.wavelengths[idx]

        self.ax_orig.clear()
        self.ax_recon.clear()

        # Get the images for current wavelength
        orig_img = self.original_image[:,:,idx]
        recon_img = self.full_reconstruction[:,:,idx].numpy()

        # Print statistics for debugging
        print(f"\nDisplaying wavelength {wavelength:.1f}nm (index {idx})")
        print("Original image stats:")
        print(f"Min: {orig_img.min():.8f}")
        print(f"Max: {orig_img.max():.8f}")
        print(f"Mean: {orig_img.mean():.8f}")
        print("\nReconstructed image stats:")
        print(f"Min: {recon_img.min():.8f}")
        print(f"Max: {recon_img.max():.8f}")
        print(f"Mean: {recon_img.mean():.8f}")

        # Try different normalization for display
        if wavelength <= 900:  # VNIR band
            print("\nApplying VNIR display scaling")
            recon_img_norm = (recon_img - recon_img.min()) / (recon_img.max() - recon_img.min() + 1e-8)
            orig_img_norm = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)

            self.ax_orig.imshow(orig_img_norm, cmap='viridis')
            self.ax_recon.imshow(recon_img_norm, cmap='viridis')
        else:  # SWIR band
            print("\nApplying SWIR display scaling")
            self.ax_orig.imshow(orig_img, cmap='viridis')
            self.ax_recon.imshow(recon_img, cmap='viridis')

        self.ax_orig.set_title(f'Original Image ({wavelength:.1f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({wavelength:.1f}nm)')
        self.ax_orig.axis('off')
        self.ax_recon.axis('off')

        self.canvas.draw()

    def on_click(self, event):
        """Handle click events to show spectrum at clicked point."""
        if event.inaxes in [self.ax_orig, self.ax_recon]:
            x, y = int(event.xdata), int(event.ydata)

            if 0 <= x < self.original_image.shape[1] and 0 <= y < self.original_image.shape[0]:
                self.ax_spectrum.clear()

                # Plot original spectrum
                self.ax_spectrum.plot(self.wavelengths,
                                    self.original_image[y,x,:],
                                    'b-', label='Original')

                # Plot reconstructed spectrum
                self.ax_spectrum.plot(self.wavelengths,
                                    self.full_reconstruction[y,x,:].numpy(),
                                    'r--', label='Reconstructed')

                self.ax_spectrum.set_xlabel('Wavelength (nm)')
                self.ax_spectrum.set_ylabel('Intensity')
                self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
                self.ax_spectrum.legend()
                self.ax_spectrum.grid(True)
                plt.ylim(0,1)
                self.canvas.draw()

    def reconstruct_image(self):
        """Reconstruct the selected image."""
        start_time = time.time()

        selected_file = self.file_combo.currentText()

        # Direct data loading
        vnir_cube = np.load(os.path.join(config.dataset_path, "VNIR_RAW", f"{selected_file}.npy"))
        swir_cube = np.load(os.path.join(config.dataset_path, "SWIR_RAW", f"{selected_file}.npy"))

        # First reshape VNIR to original dimensions
        vnir_cube = vnir_cube.reshape((215, 407, 24))

        # Select wavelengths between 800-900nm
        start_idx = int(((800 - 660) / (900 - 660)) * 24)  # Calculate index for 800nm
        end_idx = start_idx + 10  # Take 10 points from there
        vnir_cube = vnir_cube[:, :, start_idx:end_idx]

        # Reshape SWIR
        swir_cube = swir_cube.reshape((168, 211, 9))

        # Resize SWIR to match VNIR dimensions
        resized_swir = np.zeros((215, 407, 9))
        for i in range(9):
            resized_swir[:,:,i] = cv2.resize(swir_cube[:,:,i], (407, 215))

        # Combine cubes
        combined_cube = np.concatenate((vnir_cube, resized_swir), axis=2)

        # Store data
        self.original_image = combined_cube / np.max(combined_cube)
        # self.original_image = combined_cube

        self.wavelengths = np.concatenate([
            np.linspace(800, 900, 10),
            np.linspace(1100, 1700, 9)
        ])

        # Create dataset
        dataset = HyperspectralDataset(combined_cube)
        test_loader = DataLoader(dataset, batch_size=64)

        # Run reconstruction
        reconstructed_patches = []
        h, w, _ = combined_cube.shape
        superpixel_size = dataset.superpixel_size

        with torch.no_grad():
            for filtered_measurements, _ in test_loader:
                filtered_measurements = filtered_measurements
                outputs = self.model(filtered_measurements)
                reconstructed_patches.append(outputs)

        # Reshape reconstructions back into image
        reconstructed_patches = torch.cat(reconstructed_patches, dim=0)
        patches_per_row = w // superpixel_size

        # Initialize full reconstruction array
        self.full_reconstruction = torch.zeros((h, w, 19), dtype=torch.float32)

        # Place patches back into image
        for idx in range(len(reconstructed_patches)):
            i = (idx // patches_per_row) * superpixel_size
            j = (idx % patches_per_row) * superpixel_size
            self.full_reconstruction[i:i+superpixel_size, j:j+superpixel_size, :] = \
                reconstructed_patches[idx].permute(1, 2, 0)

        # Calculate RMSE
        mse = torch.mean((self.full_reconstruction - torch.tensor(self.original_image)) ** 2)
        rmse = torch.sqrt(mse)

        # Update time and RMSE labels
        elapsed_time = time.time() - start_time
        self.time_label.setText(f"Time: {elapsed_time:.2f}s")
        self.rmse_label.setText(f"RMSE: {rmse:.4f}")

        # Update display
        self.update_wavelength_display()

def main():
    app = QApplication(sys.argv)
    viewer = ReconstructionViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
