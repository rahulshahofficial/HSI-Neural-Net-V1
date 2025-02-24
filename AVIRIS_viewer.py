import sys
import numpy as np
import rasterio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                            QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider, 
                            QFileDialog, QSplitter)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import os

class HyperspectralViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AVIRIS Hyperspectral Image Viewer')
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.hypercube = None
        self.data_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_augmented_dataset_2"
        self.wavelengths = np.linspace(400, 2500, 220)  # Full wavelength range
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Create controls panel
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        
        # File selection
        self.file_combo = QComboBox()
        self.refresh_file_list()
        controls_layout.addWidget(QLabel("Image:"))
        controls_layout.addWidget(self.file_combo)
        
        # Load button
        load_btn = QPushButton("Load Image")
        load_btn.clicked.connect(self.load_image)
        controls_layout.addWidget(load_btn)
        
        # Browse button
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_files)
        controls_layout.addWidget(browse_btn)
        
        # Wavelength slider
        self.wavelength_slider = QSlider(Qt.Horizontal)
        self.wavelength_slider.setMinimum(0)
        self.wavelength_slider.setMaximum(219)  # 220 bands (0-219)
        self.wavelength_slider.setValue(100)  # Middle-ish band
        self.wavelength_slider.valueChanged.connect(self.update_display)
        controls_layout.addWidget(QLabel("Band:"))
        controls_layout.addWidget(self.wavelength_slider)
        
        # Wavelength display
        self.wavelength_label = QLabel("Wavelength: N/A")
        controls_layout.addWidget(self.wavelength_label)
        
        # Add info display
        self.info_label = QLabel("No image loaded")
        
        # Add controls to main layout
        layout.addWidget(controls)
        layout.addWidget(self.info_label)
        
        # Create a splitter for the plots
        splitter = QSplitter(Qt.Horizontal)
        
        # Image display
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        self.fig_image = plt.figure(figsize=(6, 6))
        self.ax_image = self.fig_image.add_subplot(111)
        self.canvas_image = FigureCanvas(self.fig_image)
        image_layout.addWidget(self.canvas_image)
        
        # Spectrum display
        spectrum_widget = QWidget()
        spectrum_layout = QVBoxLayout(spectrum_widget)
        self.fig_spectrum = plt.figure(figsize=(6, 6))
        self.ax_spectrum = self.fig_spectrum.add_subplot(111)
        self.canvas_spectrum = FigureCanvas(self.fig_spectrum)
        spectrum_layout.addWidget(self.canvas_spectrum)
        
        # Add widgets to splitter
        splitter.addWidget(image_widget)
        splitter.addWidget(spectrum_widget)
        
        # Add splitter to main layout
        layout.addWidget(splitter)
        
        # Connect mouse click event
        self.canvas_image.mpl_connect('button_press_event', self.on_click)
    
    def refresh_file_list(self):
        """Refresh the list of available files in the combobox"""
        self.file_combo.clear()
        try:
            files = sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tif')])
            self.file_combo.addItems(files)
        except Exception as e:
            print(f"Error loading file list: {str(e)}")
    
    def browse_files(self):
        """Open file browser to select a directory"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory", self.data_dir)
        if directory:
            self.data_dir = directory
            self.refresh_file_list()
    
    def load_image(self):
        """Load the selected hyperspectral image"""
        if self.file_combo.currentText() == "":
            return
            
        try:
            file_path = os.path.join(self.data_dir, self.file_combo.currentText())
            
            with rasterio.open(file_path) as src:
                # Read all bands and transpose to (H, W, C)
                data = src.read()
                self.hypercube = np.transpose(data, (1, 2, 0))
                
                # Update info
                h, w, c = self.hypercube.shape
                self.info_label.setText(f"Image dimensions: {h}×{w} pixels, {c} bands")
                
                # Update display
                self.update_display()
                
        except Exception as e:
            self.info_label.setText(f"Error loading image: {str(e)}")
    
    def update_display(self):
        """Update the image display when wavelength changes"""
        if self.hypercube is None:
            return

        # Get current band and wavelength
        band_idx = self.wavelength_slider.value()
        wavelength = self.wavelengths[band_idx]
        self.wavelength_label.setText(f"Wavelength: {wavelength:.1f} nm (Band {band_idx})")

        # Clear the figure completely including any colorbars
        self.fig_image.clear()
        self.ax_image = self.fig_image.add_subplot(111)

        # Update image
        band_image = self.hypercube[:, :, band_idx]
        im = self.ax_image.imshow(band_image, cmap='viridis')
        self.ax_image.set_title(f'Band {band_idx} ({wavelength:.1f} nm)')
        self.fig_image.colorbar(im, ax=self.ax_image)
        self.canvas_image.draw()

        # Reconnect mouse click event (needed after clearing figure)
        self.canvas_image.mpl_connect('button_press_event', self.on_click)
    
    def on_click(self, event):
        """Handle click events to show spectrum at clicked point"""
        if self.hypercube is None or event.inaxes != self.ax_image:
            return

        x, y = int(event.xdata), int(event.ydata)

        # Check bounds
        h, w, _ = self.hypercube.shape
        if 0 <= x < w and 0 <= y < h:
            # Get spectrum at clicked location
            spectrum = self.hypercube[y, x, :]

            # Plot spectrum
            self.ax_spectrum.clear()
            self.ax_spectrum.plot(self.wavelengths, spectrum)
            self.ax_spectrum.set_xlabel('Wavelength (nm)')
            self.ax_spectrum.set_ylabel('Intensity')
            self.ax_spectrum.set_title(f'Spectrum at Pixel ({x}, {y})')
            self.ax_spectrum.grid(True)

            # Mark non-zero regions
            regions = np.where(spectrum > 0.01)[0]
            if len(regions) > 0:
                min_idx, max_idx = regions[0], regions[-1]
                self.ax_spectrum.axvspan(self.wavelengths[min_idx], self.wavelengths[max_idx],
                                        alpha=0.2, color='green')
                self.ax_spectrum.text(0.02, 0.95, f"Active range: {self.wavelengths[min_idx]:.1f}-{self.wavelengths[max_idx]:.1f} nm",
                                    transform=self.ax_spectrum.transAxes, verticalalignment='top',
                                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

            # Draw marker directly without redrawing the whole image
            self.ax_image.plot(x, y, 'ro', markersize=5)
            
            # Update both canvases
            self.canvas_spectrum.draw()
            self.canvas_image.draw()

def main():
    app = QApplication(sys.argv)
    viewer = HyperspectralViewer()
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
