from config import config
from network import HyperspectralNet
from dataset import HyperspectralDataset
from main import HyperspectralViewer

import sys
import time
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider)
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
        self.wavelength_slider.setMaximum(32)  # 33 wavelengths - 1
        self.wavelength_slider.setValue(16)  # Middle wavelength
        self.wavelength_slider.valueChanged.connect(self.update_wavelength_display)
        control_layout.addWidget(QLabel("Wavelength:"))
        control_layout.addWidget(self.wavelength_slider)
        
        # Add wavelength display label
        self.wavelength_label = QLabel("Wavelength: N/A")
        control_layout.addWidget(self.wavelength_label)
        
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
        self.model.load_state_dict(torch.load(config.model_save_path))
        self.model.eval()
    
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
        
        self.ax_orig.clear()
        self.ax_recon.clear()
        
        self.ax_orig.imshow(self.original_image[:,:,idx])
        self.ax_orig.set_title('Original Image')
        self.ax_orig.axis('off')
        
        self.ax_recon.imshow(self.full_reconstruction[:,:,idx].numpy())
        self.ax_recon.set_title('Reconstructed Image')
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
                
                self.canvas.draw()
    
    def reconstruct_image(self):
        """Reconstruct the selected image."""
        start_time = time.time()
        
        selected_file = self.file_combo.currentText()
        
        # Get file paths
        vnir_file = os.path.join(config.dataset_path, "VNIR_RAW", f"{selected_file}.npy")
        swir_file = os.path.join(config.dataset_path, "SWIR_RAW", f"{selected_file}.npy")
        
        # Load and process image
        viewer = HyperspectralViewer(vnir_file, swir_file)
        self.original_image = viewer.combined_cube
        self.wavelengths = viewer.wavelengths
        
        dataset = HyperspectralDataset(viewer.combined_cube)
        test_loader = DataLoader(dataset, batch_size=64)
        
        # Run reconstruction
        reconstructed_patches = []
        h, w, _ = viewer.combined_cube.shape
        superpixel_size = dataset.superpixel_size
        
        with torch.no_grad():
            for filtered_measurements, _ in test_loader:
                outputs = self.model(filtered_measurements)
                reconstructed_patches.append(outputs)
        
        # Reshape reconstructions back into image
        reconstructed_patches = torch.cat(reconstructed_patches, dim=0)
        patches_per_row = w // superpixel_size
        
        # Initialize full reconstruction array
        self.full_reconstruction = torch.zeros_like(torch.tensor(viewer.combined_cube))
        
        # Place patches back into full image
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
