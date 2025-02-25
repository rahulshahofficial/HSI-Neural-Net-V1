import time
import numpy as np
import torch
import json
from datetime import datetime
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                           QHBoxLayout, QComboBox, QPushButton, QLabel, QSlider, QLineEdit,
                           QTabWidget, QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView,
                           QFileDialog, QMessageBox, QProgressBar, QGroupBox, QSplitter)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import os
import rasterio
import pandas as pd

from config import config
from srnet_model import SpectralReconstructionNet
from dataset import FullImageHyperspectralDataset
from metrics import HyperspectralMetrics
from spect_dict import SpectralDictionary

class ReconstructionViewer(QMainWindow):
    """
    Enhanced GUI application for hyperspectral image reconstruction evaluation.

    Features:
    - Interactive image reconstruction and visualization
    - Comprehensive metrics calculation and display
    - Spectral and spatial analysis tools
    - Model efficiency evaluation
    - Spectral smoothness analysis
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle('AVIRIS HSI Reconstruction Viewer (SRNet) - Enhanced Metrics')
        self.setGeometry(100, 100, 1800, 1000)

        # Initialize variables
        self.full_reconstruction = None
        self.original_image = None
        self.wavelengths = None
        self.current_wavelength_idx = None
        self.current_metrics = None
        self.current_loss_components = None

        # Set data directory based on operating system
        if os.name == 'nt':  # Windows
            self.data_dir = "V:\SimulationData\Rahul\Hyperspectral Imaging Project\HSI Data Sets\AVIRIS_augmented_dataset"
        else:  # macOS or Linux
            self.data_dir = "/Volumes/ValentineLab/SimulationData/Rahul/Hyperspectral Imaging Project/HSI Data Sets/AVIRIS_augmented_dataset"

        # Make sure the directory exists, otherwise use a default location
        if not os.path.exists(self.data_dir):
            self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)

        # Create main widget with tab layout
        main_widget = QTabWidget()
        self.setCentralWidget(main_widget)

        # Create tabs
        self.reconstruction_tab = QWidget()
        self.metrics_tab = QWidget()
        self.analysis_tab = QWidget()

        main_widget.addTab(self.reconstruction_tab, "Reconstruction Viewer")
        main_widget.addTab(self.metrics_tab, "Metrics & Performance")
        main_widget.addTab(self.analysis_tab, "Analysis Tools")

        # Setup Reconstruction Tab
        self._setup_reconstruction_tab()

        # Setup Metrics Tab
        self._setup_metrics_tab()

        # Setup Analysis Tab
        self._setup_analysis_tab()

        # Load the trained model
        self.model = SpectralReconstructionNet(
            input_channels=1,
            out_channels=len(config.wavelength_indices),
            dim=64,
            deep_stage=3,
            num_blocks=[1, 2, 3],
            num_heads=[2, 4, 8],
            use_spectral_dict=True  # Enable spectral dictionary for smoother reconstruction
        )
        self.model.load_state_dict(torch.load(config.model_save_path, map_location='cpu'))
        self.model.eval()

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Initialize band comparison dropdown values after model is loaded
        for i in range(len(config.wavelength_indices)):
            wavelength = config.full_wavelengths[config.wavelength_indices[i]]
            self.band1_combo.addItem(f"{wavelength:.1f} nm")
            self.band2_combo.addItem(f"{wavelength:.1f} nm")

        # Add default values
        self.band1_combo.setCurrentIndex(0)
        self.band2_combo.setCurrentIndex(len(config.wavelength_indices) // 2)

        # Connect click events
        self.canvas.mpl_connect('button_press_event', self.on_click)

        # Analyze model efficiency
        self._analyze_model_efficiency()

        # Create results directory if it doesn't exist
        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path, exist_ok=True)

    def _setup_reconstruction_tab(self):
        """Set up the reconstruction visualization tab."""
        recon_layout = QVBoxLayout(self.reconstruction_tab)

        # Create control panel for reconstruction tab
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

        # Add plot spectrum button
        self.plot_spectrum_btn = QPushButton("Plot Spectrum")
        self.plot_spectrum_btn.clicked.connect(self.plot_coordinates)
        control_layout.addWidget(self.plot_spectrum_btn)

        # Add time display
        self.time_label = QLabel("Time: N/A")
        control_layout.addWidget(self.time_label)

        recon_layout.addWidget(control_panel)

        # Create matplotlib figures for reconstruction tab
        self.figure = plt.figure(figsize=(16, 8))
        self.gs = self.figure.add_gridspec(2, 2)
        self.ax_orig = self.figure.add_subplot(self.gs[0, 0])
        self.ax_recon = self.figure.add_subplot(self.gs[0, 1])
        self.ax_spectrum = self.figure.add_subplot(self.gs[1, :])
        self.canvas = FigureCanvas(self.figure)
        recon_layout.addWidget(self.canvas)

        # Add save button for current visualization
        save_layout = QHBoxLayout()
        self.save_visualization_btn = QPushButton("Save Current Visualization")
        self.save_visualization_btn.clicked.connect(self.save_current_visualization)
        save_layout.addWidget(self.save_visualization_btn)

        # Add export metrics button
        self.export_metrics_btn = QPushButton("Export Metrics")
        self.export_metrics_btn.clicked.connect(self.export_current_metrics)
        save_layout.addWidget(self.export_metrics_btn)

        # Add save results to layout
        recon_layout.addLayout(save_layout)

    def _setup_metrics_tab(self):
        """Set up the metrics and performance analysis tab."""
        metrics_layout = QVBoxLayout(self.metrics_tab)

        # Add a title
        title_label = QLabel("Quantitative Reconstruction Metrics")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        metrics_layout.addWidget(title_label)

        # Create two columns for metrics display
        metrics_section = QWidget()
        metrics_grid = QGridLayout(metrics_section)

        # Left column: Quality metrics
        quality_group = QGroupBox("Reconstruction Quality Metrics")
        quality_layout = QGridLayout(quality_group)

        # Create quality metrics labels
        self.metrics_labels = {}
        quality_metrics = [
            ("PSNR", "N/A dB", "Peak Signal-to-Noise Ratio - Higher is better"),
            ("SSIM", "N/A", "Structural Similarity Index - Higher is better"),
            ("RMSE", "N/A", "Root Mean Square Error - Lower is better"),
            ("MRAE", "N/A", "Mean Relative Average Error - Lower is better"),
            ("Spectral Fidelity", "N/A", "Similarity between spectral signatures - Higher is better"),
            ("SAM", "N/A", "Spectral Angle Mapper - Lower is better")
        ]

        # Add metrics to the layout with descriptions
        for i, (metric_name, default_value, tooltip) in enumerate(quality_metrics):
            # Label for metric name
            name_label = QLabel(f"{metric_name}:")
            name_label.setToolTip(tooltip)
            quality_layout.addWidget(name_label, i, 0)

            # Label for metric value
            self.metrics_labels[metric_name] = QLabel(default_value)
            self.metrics_labels[metric_name].setStyleSheet("font-weight: bold;")
            self.metrics_labels[metric_name].setToolTip(tooltip)
            quality_layout.addWidget(self.metrics_labels[metric_name], i, 1)

        metrics_grid.addWidget(quality_group, 0, 0)

        # Right column: Efficiency metrics
        efficiency_group = QGroupBox("Model Efficiency Metrics")
        efficiency_layout = QVBoxLayout(efficiency_group)

        # Create model efficiency table
        self.efficiency_table = QTableWidget(3, 2)
        self.efficiency_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.efficiency_table.verticalHeader().setVisible(False)
        self.efficiency_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        self.efficiency_table.setItem(0, 0, QTableWidgetItem("Parameters"))
        self.efficiency_table.setItem(1, 0, QTableWidgetItem("FLOPs"))
        self.efficiency_table.setItem(2, 0, QTableWidgetItem("Inference Time"))

        self.efficiency_table.setItem(0, 1, QTableWidgetItem("Calculating..."))
        self.efficiency_table.setItem(1, 1, QTableWidgetItem("Calculating..."))
        self.efficiency_table.setItem(2, 1, QTableWidgetItem("Calculating..."))

        efficiency_layout.addWidget(self.efficiency_table)

        # Add configuration info
        config_label = QLabel(f"Model Configuration: {config.num_filters} filters, {config.superpixel_height}x{config.superpixel_width} superpixels")
        config_label.setStyleSheet("font-style: italic;")
        efficiency_layout.addWidget(config_label)

        metrics_grid.addWidget(efficiency_group, 0, 1)
        metrics_layout.addWidget(metrics_section)

        # Metrics visualization section
        viz_label = QLabel("Metrics Visualization")
        viz_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        metrics_layout.addWidget(viz_label)

        self.metrics_figure = plt.figure(figsize=(12, 6))
        self.metrics_canvas = FigureCanvas(self.metrics_figure)
        metrics_layout.addWidget(self.metrics_canvas)

        # Batch evaluation section
        batch_group = QGroupBox("Batch Evaluation")
        batch_layout = QHBoxLayout(batch_group)

        self.batch_dir_btn = QPushButton("Select Directory")
        self.batch_dir_btn.clicked.connect(self.select_batch_directory)
        batch_layout.addWidget(self.batch_dir_btn)

        self.batch_evaluate_btn = QPushButton("Evaluate All Images")
        self.batch_evaluate_btn.clicked.connect(self.batch_evaluate)
        batch_layout.addWidget(self.batch_evaluate_btn)

        self.batch_progress = QProgressBar()
        self.batch_progress.setVisible(False)
        batch_layout.addWidget(self.batch_progress)

        metrics_layout.addWidget(batch_group)

    def _setup_analysis_tab(self):
        """Set up the analysis tools tab."""
        analysis_layout = QVBoxLayout(self.analysis_tab)

        # Add a title
        title_label = QLabel("Advanced Analysis Tools")
        title_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        analysis_layout.addWidget(title_label)

        # Add analysis tools
        tools_section = QWidget()
        tools_layout = QHBoxLayout(tools_section)

        # Band comparison tool
        band_comparison = QGroupBox("Spectral Band Comparison")
        band_layout = QHBoxLayout(band_comparison)

        band_layout.addWidget(QLabel("Band 1:"))
        self.band1_combo = QComboBox()
        band_layout.addWidget(self.band1_combo)

        band_layout.addWidget(QLabel("Band 2:"))
        self.band2_combo = QComboBox()
        band_layout.addWidget(self.band2_combo)

        self.compare_bands_btn = QPushButton("Compare Bands")
        self.compare_bands_btn.clicked.connect(self.compare_spectral_bands)
        band_layout.addWidget(self.compare_bands_btn)

        tools_layout.addWidget(band_comparison)

        # Error analysis tools
        error_analysis = QGroupBox("Error Analysis")
        error_layout = QHBoxLayout(error_analysis)

        self.error_map_btn = QPushButton("RMSE Map")
        self.error_map_btn.clicked.connect(lambda: self.show_error_map("rmse"))
        error_layout.addWidget(self.error_map_btn)

        self.mae_map_btn = QPushButton("MAE Map")
        self.mae_map_btn.clicked.connect(lambda: self.show_error_map("mae"))
        error_layout.addWidget(self.mae_map_btn)

        self.sam_map_btn = QPushButton("Spectral Angle Map")
        self.sam_map_btn.clicked.connect(self.show_spectral_angle_map)
        error_layout.addWidget(self.sam_map_btn)

        # Add spectral smoothness analysis button
        self.smoothness_btn = QPushButton("Analyze Spectral Smoothness")
        self.smoothness_btn.clicked.connect(self.analyze_spectral_smoothness)
        error_layout.addWidget(self.smoothness_btn)

        tools_layout.addWidget(error_analysis)

        analysis_layout.addWidget(tools_section)

        # Analysis visualization
        self.analysis_figure = plt.figure(figsize=(12, 8))
        self.analysis_canvas = FigureCanvas(self.analysis_figure)
        analysis_layout.addWidget(self.analysis_canvas)

        # Results section
        results_section = QWidget()
        results_layout = QHBoxLayout(results_section)

        self.save_analysis_btn = QPushButton("Save Analysis Results")
        self.save_analysis_btn.clicked.connect(self.save_analysis_results)
        results_layout.addWidget(self.save_analysis_btn)

        self.export_analysis_btn = QPushButton("Export Analysis Data")
        self.export_analysis_btn.clicked.connect(self.export_analysis_data)
        results_layout.addWidget(self.export_analysis_btn)

        analysis_layout.addWidget(results_section)

    def _get_available_files(self):
        """Get list of files from augmented dataset directory."""
        try:
            return sorted([f for f in os.listdir(self.data_dir) if f.endswith('.tif')])
        except:
            QMessageBox.warning(self, "Warning", f"Could not access data directory: {self.data_dir}")
            return ["sample.tif"]

    def _analyze_model_efficiency(self):
        """Analyze and display model efficiency metrics."""
        try:
            # Get sample input dimensions
            sample_input = torch.zeros((1, 1, 100, 100)).to(self.device)
            sample_filter = torch.zeros((1, len(config.wavelength_indices), 100, 100)).to(self.device)

            # Calculate efficiency metrics
            efficiency_metrics = HyperspectralMetrics.analyze_model_efficiency(
                self.model, sample_input.shape, sample_filter.shape, device=self.device
            )

            # Update efficiency table
            num_params = efficiency_metrics['num_params']
            param_str = f"{num_params:,}" if num_params < 1e6 else f"{num_params/1e6:.2f}M"
            self.efficiency_table.setItem(0, 1, QTableWidgetItem(param_str))

            flops = efficiency_metrics['flops']
            if flops > 1e9:
                flops_str = f"{flops/1e9:.2f} GFLOPs"
            else:
                flops_str = f"{flops/1e6:.2f} MFLOPs"
            self.efficiency_table.setItem(1, 1, QTableWidgetItem(flops_str))

            time_ms = efficiency_metrics['time_per_frame'] * 1000
            self.efficiency_table.setItem(2, 1, QTableWidgetItem(f"{time_ms:.2f} ms"))

            # Save to file for future reference
            os.makedirs('results', exist_ok=True)
            efficiency_data = {
                'parameters': num_params,
                'flops': flops,
                'time_per_frame_ms': time_ms,
                'model_config': {
                    'dimension': 64,
                    'stages': 3,
                    'blocks': [1, 2, 3],
                    'heads': [2, 4, 8]
                }
            }

            # Save to file (append if exists)
            try:
                if os.path.exists('results/efficiency_history.json'):
                    with open('results/efficiency_history.json', 'r') as f:
                        history = json.load(f)
                    history.append({
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'data': efficiency_data
                    })
                else:
                    history = [{
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'data': efficiency_data
                    }]

                with open('results/efficiency_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
            except:
                print("Warning: Could not save efficiency metrics to file")
        except Exception as e:
            print(f"Error analyzing model efficiency: {str(e)}")
            self.efficiency_table.setItem(0, 1, QTableWidgetItem("Error"))
            self.efficiency_table.setItem(1, 1, QTableWidgetItem("Error"))
            self.efficiency_table.setItem(2, 1, QTableWidgetItem("Error"))

    def reconstruct_image(self):
        """Reconstruct the selected image using SRNet and calculate comprehensive metrics."""
        if not self.file_combo.currentText():
            QMessageBox.warning(self, "Warning", "No file selected.")
            return

        start_time = time.time()
        selected_file = self.file_combo.currentText()

        try:
            # Load hyperspectral data
            with rasterio.open(os.path.join(self.data_dir, selected_file)) as src:
                img_data = src.read()
                img_data = np.transpose(img_data, (1, 2, 0))  # Change to (H,W,C)
                img_data = img_data.reshape((1, *img_data.shape))  # Add batch dimension

            # Create dataset with single image
            dataset = FullImageHyperspectralDataset(img_data)
            # Get the filtered measurements, filter pattern, and original spectrum
            filtered_measurements, filter_pattern, original_spectrum = dataset[0]

            # Store wavelengths and original image
            self.wavelengths = config.full_wavelengths[config.wavelength_indices]
            self.original_image = original_spectrum.numpy()

            # Move to appropriate device
            filtered_measurements = filtered_measurements.to(self.device)
            filter_pattern = filter_pattern.to(self.device)

            # Perform reconstruction
            with torch.no_grad():
                # Add batch dimension
                filtered_measurements = filtered_measurements.unsqueeze(0)
                filter_pattern = filter_pattern.unsqueeze(0)

                # Run model
                reconstructed = self.model(filtered_measurements, filter_pattern)

                # Move back to CPU and remove batch dimension
                self.full_reconstruction = reconstructed.cpu().squeeze(0)

            # Calculate comprehensive metrics
            metrics = HyperspectralMetrics.compute_all_metrics(
                self.full_reconstruction, torch.tensor(self.original_image)
            )

            # Calculate combined loss components
            with torch.no_grad():
                reconstructed_batch = self.full_reconstruction.unsqueeze(0)
                original_batch = torch.tensor(self.original_image).unsqueeze(0)
                criterion = torch.nn.MSELoss()

                _, loss_components = self.model.compute_loss(
                    reconstructed_batch, original_batch, criterion
                )

            # Update time label
            elapsed_time = time.time() - start_time
            self.time_label.setText(f"Time: {elapsed_time:.2f}s")

            # Update metrics display
            self.metrics_labels["PSNR"].setText(f"{metrics['psnr']:.2f} dB")
            self.metrics_labels["SSIM"].setText(f"{metrics['ssim']:.4f}")
            self.metrics_labels["RMSE"].setText(f"{metrics['rmse']:.6f}")
            self.metrics_labels["MRAE"].setText(f"{metrics['mrae']:.4f}")
            self.metrics_labels["Spectral Fidelity"].setText(f"{metrics['spectral_fidelity']:.4f}")

            # Calculate spectral angle mapper (SAM)
            sam = loss_components.get('spectral_angle_loss', 0.0)
            self.metrics_labels["SAM"].setText(f"{sam:.4f}")

            # Update metrics visualization
            self._update_metrics_visualization(metrics, loss_components)

            # Update main display
            self.update_wavelength_display()

            # Store metrics for later use
            self.current_metrics = metrics
            self.current_loss_components = loss_components

            # Set window title with current file
            self.setWindowTitle(f'HSI Reconstruction - {selected_file} - PSNR: {metrics["psnr"]:.2f}dB')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to reconstruct image: {str(e)}")
            print(f"Error: {str(e)}")

    def update_wavelength_display(self):
        """Update display when wavelength slider changes."""
        if self.wavelengths is None or self.original_image is None:
            return

        idx = self.wavelength_slider.value()
        self.current_wavelength_idx = idx
        wavelength = self.wavelengths[idx]
        self.wavelength_label.setText(f"Wavelength: {wavelength:.2f} nm")

        # Update images
        self.ax_orig.clear()
        self.ax_recon.clear()

        orig_img = self.original_image[idx]
        recon_img = self.full_reconstruction[idx].numpy()

        # Use same color scale for fair comparison
        vmin = min(orig_img.min(), recon_img.min())
        vmax = max(orig_img.max(), recon_img.max())

        orig_plot = self.ax_orig.imshow(orig_img, cmap='viridis', vmin=vmin, vmax=vmax)
        recon_plot = self.ax_recon.imshow(recon_img, cmap='viridis', vmin=vmin, vmax=vmax)

        self.ax_orig.set_title(f'Original Image ({wavelength:.1f}nm)')
        self.ax_recon.set_title(f'Reconstructed Image ({wavelength:.1f}nm)')
        self.ax_orig.axis('off')
        self.ax_recon.axis('off')

        # Add colorbars
        self.figure.colorbar(orig_plot, ax=self.ax_orig, fraction=0.046, pad=0.04)
        self.figure.colorbar(recon_plot, ax=self.ax_recon, fraction=0.046, pad=0.04)

        # If no spectrum has been plotted yet, clear the spectrum axis
        if len(self.ax_spectrum.lines) == 0:
            self.ax_spectrum.clear()
            self.ax_spectrum.set_title('Click on image to see spectrum')
            self.ax_spectrum.set_xlabel('Wavelength (nm)')
            self.ax_spectrum.set_ylabel('Intensity')
            self.ax_spectrum.grid(True)

        self.figure.tight_layout()
        self.canvas.draw()

    def plot_coordinates(self):
        """Plot spectrum for manually entered coordinates."""
        try:
            x = int(self.x_coord.text())
            y = int(self.y_coord.text())
            self.plot_spectrum(x, y)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter valid integer coordinates.")
    def plot_spectrum(self, x, y):
        """Plot spectrum for given coordinates."""
        if self.original_image is None or self.full_reconstruction is None:
            return

        h, w = self.original_image.shape[1:]
        if 0 <= y < h and 0 <= x < w:
            self.ax_spectrum.clear()

            # Plot original and reconstructed spectra
            self.ax_spectrum.plot(self.wavelengths,
                                self.original_image[:, y, x],
                                'b-', label='Original', linewidth=2)
            self.ax_spectrum.plot(self.wavelengths,
                                self.full_reconstruction[:, y, x].numpy(),
                                'r--', label='Reconstructed', linewidth=2)

            # Mark current wavelength if available
            if self.current_wavelength_idx is not None:
                current_wl = self.wavelengths[self.current_wavelength_idx]
                orig_value = self.original_image[self.current_wavelength_idx, y, x]
                recon_value = self.full_reconstruction[self.current_wavelength_idx, y, x].numpy()

                self.ax_spectrum.axvline(current_wl, color='gray', linestyle=':', alpha=0.7)
                self.ax_spectrum.plot([current_wl], [orig_value], 'bo', markersize=8)
                self.ax_spectrum.plot([current_wl], [recon_value], 'ro', markersize=8)

            # Set labels and title
            self.ax_spectrum.set_xlabel('Wavelength (nm)')
            self.ax_spectrum.set_ylabel('Intensity')
            self.ax_spectrum.set_title(f'Spectrum at ({x}, {y})')
            self.ax_spectrum.legend()
            self.ax_spectrum.grid(True)

            # Calculate local metrics
            orig_spectrum = self.original_image[:, y, x]
            recon_spectrum = self.full_reconstruction[:, y, x].numpy()

            # Calculate spectral angle
            norm_orig = orig_spectrum / np.linalg.norm(orig_spectrum)
            norm_recon = recon_spectrum / np.linalg.norm(recon_spectrum)
            dot_product = np.dot(norm_orig, norm_recon)
            dot_product = np.clip(dot_product, -1.0, 1.0)  # Clip to prevent numerical errors
            spectral_angle = np.arccos(dot_product)

            # Add metrics to the plot
            metrics_text = (
                f"Point Metrics at ({x},{y}):\n"
                f"RMSE: {np.sqrt(np.mean((orig_spectrum - recon_spectrum)**2)):.4f}\n"
                f"Spectral Angle: {spectral_angle:.4f} rad\n"
                f"Correlation: {np.corrcoef(orig_spectrum, recon_spectrum)[0,1]:.4f}"
            )

            # Place the text in the upper right corner
            self.ax_spectrum.text(
                0.98, 0.98, metrics_text,
                transform=self.ax_spectrum.transAxes,
                ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )

            # Set y-axis limits with some padding
            min_val = min(orig_spectrum.min(), recon_spectrum.min())
            max_val = max(orig_spectrum.max(), recon_spectrum.max())
            padding = 0.1 * (max_val - min_val)
            self.ax_spectrum.set_ylim(min_val - padding, max_val + padding)

            self.canvas.draw()

    def on_click(self, event):
        """Handle click events to show spectrum at clicked point."""
        if event.inaxes in [self.ax_orig, self.ax_recon]:
            x, y = int(event.xdata), int(event.ydata)
            # Update coordinate input boxes
            self.x_coord.setText(str(x))
            self.y_coord.setText(str(y))
            self.plot_spectrum(x, y)

    def _update_metrics_visualization(self, metrics, loss_components):
        """Update the metrics visualization tab with current results."""
        self.metrics_figure.clear()

        # Create bar chart of key metrics
        ax1 = self.metrics_figure.add_subplot(121)
        metric_names = ['PSNR/50', 'SSIM', 'Spectral Fidelity']
        metric_values = [metrics['psnr']/50, metrics['ssim'], metrics['spectral_fidelity']]  # Normalize PSNR

        bars = ax1.bar(metric_names, metric_values, color=['green', 'blue', 'purple'])
        ax1.set_ylim(0, 1.0)
        ax1.set_title('Quality Metrics (Higher is Better)')

        # Add value labels
        for bar, name, value in zip(bars, metric_names, [metrics['psnr'], metrics['ssim'], metrics['spectral_fidelity']]):
            if name == 'PSNR/50':
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{value:.2f} dB", ha='center', va='bottom')
            else:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                         f"{value:.4f}", ha='center', va='bottom')

        # Add error metrics (lower is better)
        ax2 = self.metrics_figure.add_subplot(122)
        error_names = ['RMSE', 'MRAE', 'SAM']
        error_values = [
            metrics['rmse'],
            metrics['mrae'],
            loss_components.get('spectral_angle_loss', 0)
        ]

        bars = ax2.bar(error_names, error_values, color=['red', 'orange', 'brown'])
        ax2.set_title('Error Metrics (Lower is Better)')

        for bar, value in zip(bars, error_values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{value:.4f}", ha='center', va='bottom')

        self.metrics_figure.tight_layout()
        self.metrics_canvas.draw()

    def compare_spectral_bands(self):
        """Compare reconstruction quality between two spectral bands."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Get selected band indices
        band1_idx = self.band1_combo.currentIndex()
        band2_idx = self.band2_combo.currentIndex()

        # Clear previous figure
        self.analysis_figure.clear()

        # Plot comparison
        ax1 = self.analysis_figure.add_subplot(221)
        ax2 = self.analysis_figure.add_subplot(222)
        ax3 = self.analysis_figure.add_subplot(223)
        ax4 = self.analysis_figure.add_subplot(224)

        # Original vs Reconstructed for band 1
        orig_band1 = self.original_image[band1_idx]
        recon_band1 = self.full_reconstruction[band1_idx].numpy()

        ax1.imshow(orig_band1, cmap='viridis')
        ax1.set_title(f'Original Band {band1_idx} ({self.wavelengths[band1_idx]:.1f}nm)')
        ax1.axis('off')

        ax2.imshow(recon_band1, cmap='viridis')
        ax2.set_title(f'Reconstructed Band {band1_idx}')
        ax2.axis('off')

        # Original vs Reconstructed for band 2
        orig_band2 = self.original_image[band2_idx]
        recon_band2 = self.full_reconstruction[band2_idx].numpy()

        ax3.imshow(orig_band2, cmap='viridis')
        ax3.set_title(f'Original Band {band2_idx} ({self.wavelengths[band2_idx]:.1f}nm)')
        ax3.axis('off')

        ax4.imshow(recon_band2, cmap='viridis')
        ax4.set_title(f'Reconstructed Band {band2_idx}')
        ax4.axis('off')

        # Calculate per-band metrics
        rmse1 = np.sqrt(np.mean((orig_band1 - recon_band1) ** 2))
        rmse2 = np.sqrt(np.mean((orig_band2 - recon_band2) ** 2))

        # Add metrics as text
        ax2.text(0.5, -0.1, f'RMSE: {rmse1:.4f}', ha='center', va='center',
                transform=ax2.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        ax4.text(0.5, -0.1, f'RMSE: {rmse2:.4f}', ha='center', va='center',
                transform=ax4.transAxes, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def show_error_map(self, error_type="rmse"):
        """Show spatial error distribution map."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Select current wavelength for analysis
        wavelength_idx = self.current_wavelength_idx if self.current_wavelength_idx is not None else 0
        wavelength = self.wavelengths[wavelength_idx]

        orig = self.original_image[wavelength_idx]
        recon = self.full_reconstruction[wavelength_idx].numpy()

        # Create error maps
        if error_type == "rmse":
            error_map = np.abs(orig - recon)
            title = f'Absolute Error Map at {wavelength:.1f}nm'
            cmap = 'hot'
        elif error_type == "mae":
            error_map = np.abs(orig - recon)
            title = f'Mean Absolute Error Map at {wavelength:.1f}nm'
            cmap = 'hot'
        else:
            error_map = np.abs(orig - recon) / (np.abs(orig) + 1e-6)
            title = f'Relative Error Map at {wavelength:.1f}nm'
            cmap = 'hot'

        # Create subplots
        ax1 = self.analysis_figure.add_subplot(131)
        ax2 = self.analysis_figure.add_subplot(132)
        ax3 = self.analysis_figure.add_subplot(133)

        # Plot original and reconstructed images
        ax1.imshow(orig, cmap='viridis')
        ax1.set_title(f'Original at {wavelength:.1f}nm')
        ax1.axis('off')

        ax2.imshow(recon, cmap='viridis')
        ax2.set_title(f'Reconstructed at {wavelength:.1f}nm')
        ax2.axis('off')

        # Plot error map
        im = ax3.imshow(error_map, cmap=cmap)
        ax3.set_title(title)
        ax3.axis('off')
        self.analysis_figure.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

        # Add overall metrics
        if error_type == "rmse":
            global_error = np.sqrt(np.mean((orig - recon) ** 2))
            metric_name = "RMSE"
        elif error_type == "mae":
            global_error = np.mean(np.abs(orig - recon))
            metric_name = "MAE"
        else:
            global_error = np.mean(np.abs(orig - recon) / (np.abs(orig) + 1e-6))
            metric_name = "MRAE"

        ax3.text(0.5, -0.1, f'Global {metric_name}: {global_error:.4f}',
                ha='center', va='center', transform=ax3.transAxes,
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

        self.analysis_figure.tight_layout()
        self.analysis_canvas.draw()

    def show_spectral_angle_map(self):
        """Show spectral angle mapper (SAM) visualization."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Calculate Spectral Angle Map
        orig = self.original_image  # Shape: [C, H, W]
        recon = self.full_reconstruction.numpy()  # Shape: [C, H, W]

        # Reshape for easier calculations
        c, h, w = orig.shape
        orig_reshaped = orig.reshape(c, -1)  # [C, H*W]
        recon_reshaped = recon.reshape(c, -1)  # [C, H*W]

        # Normalize each spectral vector
        orig_norm = np.linalg.norm(orig_reshaped, axis=0, keepdims=True)
        recon_norm = np.linalg.norm(recon_reshaped, axis=0, keepdims=True)

        # Avoid division by zero
        orig_normalized = orig_reshaped / (orig_norm + 1e-8)
        recon_normalized = recon_reshaped / (recon_norm + 1e-8)

        # Calculate dot product and clip to valid range
        dot_product = np.sum(orig_normalized * recon_normalized, axis=0)
        dot_product = np.clip(dot_product, -1.0, 1.0)

        # Calculate spectral angle
        sam_map = np.arccos(dot_product)
        sam_map = sam_map.reshape(h, w)

        # Plot results
        ax1 = self.analysis_figure.add_subplot(121)
        im = ax1.imshow(sam_map, cmap='hot')
        ax1.set_title('Spectral Angle Map (SAM)')
        ax1.axis('off')
        self.analysis_figure.colorbar(im, ax=ax1, label='Angle (radians)')

        # Plot histogram of SAM values
        ax2 = self.analysis_figure.add_subplot(122)
        ax2.hist(sam_map.flatten(), bins=50, color='red', alpha=0.7)
        ax2.set_title('Distribution of Spectral Angles')
        ax2.set_xlabel('Spectral Angle (radians)')
        ax2.set_ylabel('Frequency')

        # Add mean SAM value
        mean_sam = np.mean(sam_map)
        ax2.axvline(mean_sam, color='black', linestyle='--',
                   label=f'Mean: {mean_sam:.4f} rad')
        ax2.legend()

        # Add global metric to figure
        self.analysis_figure.suptitle(
            f'Spectral Angle Analysis - Mean SAM: {mean_sam:.4f} rad',
            fontsize=14
        )

        self.analysis_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.analysis_canvas.draw()

    def analyze_spectral_smoothness(self):
        """Analyze and visualize the spectral smoothness improvement."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "Please reconstruct an image first.")
            return

        # Clear previous figure
        self.analysis_figure.clear()

        # Pick a few random pixels to analyze
        h, w = self.original_image.shape[1:]
        num_pixels = 4
        pixels = []

        # Try to find pixels with interesting spectra
        for _ in range(20):  # Try up to 20 random positions to find good examples
            if len(pixels) >= num_pixels:
                break

            x, y = np.random.randint(0, w), np.random.randint(0, h)
            orig_spectrum = self.original_image[:, y, x]

            # Check if spectrum has enough variation (not just flat)
            if np.std(orig_spectrum) > 0.05:
                pixels.append((x, y))

        # If we couldn't find enough interesting pixels, just pick random ones
        while len(pixels) < num_pixels:
            x, y = np.random.randint(0, w), np.random.randint(0, h)
            pixels.append((x, y))

        # Calculate spectral derivatives for each pixel
        for i, (x, y) in enumerate(pixels):
            # Get spectra
            orig_spectrum = self.original_image[:, y, x]
            recon_spectrum = self.full_reconstruction[:, y, x].numpy()

            # Create subplot
            ax_spec = self.analysis_figure.add_subplot(num_pixels, 2, i*2 + 1)
            ax_deriv = self.analysis_figure.add_subplot(num_pixels, 2, i*2 + 2)

            # Plot spectra
            ax_spec.plot(self.wavelengths, orig_spectrum, 'b-', label='Original', linewidth=2)
            ax_spec.plot(self.wavelengths, recon_spectrum, 'r-', label='Reconstructed', linewidth=2)
            ax_spec.set_title(f'Spectrum at ({x}, {y})')
            if i == 0:
                ax_spec.legend()

            # Calculate and plot derivatives (for smoothness analysis)
            orig_deriv = np.diff(orig_spectrum)
            recon_deriv = np.diff(recon_spectrum)

            derivative_x = self.wavelengths[1:]  # x-axis for derivatives

            ax_deriv.plot(derivative_x, orig_deriv, 'b-', label='Original Derivative', alpha=0.7)
            ax_deriv.plot(derivative_x, recon_deriv, 'r-', label='Reconstructed Derivative', alpha=0.7)
            ax_deriv.set_title('Spectral Derivatives (Smoothness)')
            if i == 0:
                ax_deriv.legend()

            # Calculate smoothness metrics
            orig_roughness = np.sqrt(np.mean(orig_deriv**2))
            recon_roughness = np.sqrt(np.mean(recon_deriv**2))

            # Add roughness values to the plot
            ax_deriv.text(0.05, 0.95, f"Original roughness: {orig_roughness:.4f}",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.7))
            ax_deriv.text(0.05, 0.85, f"Reconstructed roughness: {recon_roughness:.4f}",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     bbox=dict(facecolor='white', alpha=0.7))

            # If the reconstructed is smoother, highlight it
            if recon_roughness < orig_roughness:
                ax_deriv.text(0.05, 0.75, "✓ Reconstruction is smoother",
                     transform=ax_deriv.transAxes, va='top', ha='left',
                     color='green', fontweight='bold',
                     bbox=dict(facecolor='white', alpha=0.7))

        # Add overall title
        self.analysis_figure.suptitle("Spectral Smoothness Analysis", fontsize=14)
        self.analysis_figure.tight_layout(rect=[0, 0, 1, 0.95])
        self.analysis_canvas.draw()

    def save_current_visualization(self):
        """Save the current visualization to a file."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "No reconstruction to save.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization",
            os.path.join(config.results_path, "visualization.png"),
            "PNG Files (*.png);;All Files (*)"
        )

        if not file_path:
            return

        # Save figure
        self.figure.savefig(file_path, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Success", f"Visualization saved to {file_path}")

    def export_current_metrics(self):
        """Export the current metrics to a CSV file."""
        if not hasattr(self, 'current_metrics') or self.current_metrics is None:
            QMessageBox.warning(self, "Warning", "No metrics to export.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Metrics",
            os.path.join(config.results_path, "metrics.csv"),
            "CSV Files (*.csv);;All Files (*)"
        )

        if not file_path:
            return

        # Combine all metrics
        all_metrics = {
            'filename': self.file_combo.currentText(),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            **self.current_metrics,
            **{f'loss_{k}': v for k, v in self.current_loss_components.items()},
            'inference_time_ms': float(self.time_label.text().split(':')[1].strip('s')) * 1000
        }

        # Write to CSV
        df = pd.DataFrame([all_metrics])
        df.to_csv(file_path, index=False)
        QMessageBox.information(self, "Success", f"Metrics exported to {file_path}")

    def save_analysis_results(self):
        """Save the current analysis visualization to a file."""
        if len(self.analysis_figure.axes) == 0:
            QMessageBox.warning(self, "Warning", "No analysis to save.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Analysis",
            os.path.join(config.results_path, "analysis.png"),
            "PNG Files (*.png);;All Files (*)"
        )

        if not file_path:
            return

        # Save figure
        self.analysis_figure.savefig(file_path, dpi=300, bbox_inches='tight')
        QMessageBox.information(self, "Success", f"Analysis saved to {file_path}")

    def export_analysis_data(self):
        """Export the current analysis data to a file."""
        if not hasattr(self, 'full_reconstruction') or self.full_reconstruction is None:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        # Get file name
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Analysis Data",
            os.path.join(config.results_path, "analysis_data.npz"),
            "NumPy Files (*.npz);;All Files (*)"
        )

        if not file_path:
            return

        # Export data
        np.savez(
            file_path,
            original=self.original_image,
            reconstructed=self.full_reconstruction.numpy(),
            wavelengths=self.wavelengths,
            metrics=self.current_metrics,
            filename=self.file_combo.currentText()
        )

        QMessageBox.information(self, "Success", f"Analysis data exported to {file_path}")

    def select_batch_directory(self):
        """Select directory for batch processing."""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Directory for Batch Processing",
            self.data_dir
        )

        if dir_path:
            self.batch_dir_path = dir_path
            # Count TIF files in the directory
            tif_files = [f for f in os.listdir(dir_path) if f.lower().endswith(('.tif', '.tiff'))]
            QMessageBox.information(
                self, "Selected Directory",
                f"Selected directory: {dir_path}\nFound {len(tif_files)} TIF files"
            )

    def batch_evaluate(self):
        """Evaluate all images in the selected directory."""
        if not hasattr(self, 'batch_dir_path'):
            QMessageBox.warning(self, "Warning", "Please select a directory first.")
            return

        # Find all TIF files
        tif_files = [f for f in os.listdir(self.batch_dir_path)
                   if f.lower().endswith(('.tif', '.tiff'))]

        if not tif_files:
            QMessageBox.warning(self, "Warning", "No TIF files found in the selected directory.")
            return

        # Setup progress bar
        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(len(tif_files))
        self.batch_progress.setValue(0)

        # Create results dataframe
        results = []

        # Process each file
        for i, file_name in enumerate(tif_files):
            try:
                # Update progress
                self.batch_progress.setValue(i)
                QApplication.processEvents()  # Keep UI responsive

                # Load hyperspectral data
                file_path = os.path.join(self.batch_dir_path, file_name)
                with rasterio.open(file_path) as src:
                    img_data = src.read()
                    img_data = np.transpose(img_data, (1, 2, 0))
                    img_data = img_data.reshape((1, *img_data.shape))

                # Create dataset
                dataset = FullImageHyperspectralDataset(img_data)
                filtered_measurements, filter_pattern, original_spectrum = dataset[0]

                # Measure inference time
                start_time = time.time()

                # Perform reconstruction
                with torch.no_grad():
                    filtered_measurements = filtered_measurements.to(self.device).unsqueeze(0)
                    filter_pattern = filter_pattern.to(self.device).unsqueeze(0)
                    reconstructed = self.model(filtered_measurements, filter_pattern)
                    reconstructed = reconstructed.cpu().squeeze(0)

                # Calculate inference time
                inference_time = time.time() - start_time

                # Calculate metrics
                metrics = HyperspectralMetrics.compute_all_metrics(
                    reconstructed, original_spectrum
                )

                # Calculate loss components
                with torch.no_grad():
                    _, loss_components = self.model.compute_loss(
                        reconstructed.unsqueeze(0),
                        original_spectrum.unsqueeze(0),
                        torch.nn.MSELoss()
                    )

                # Store results
                result = {
                    'filename': file_name,
                    'inference_time_ms': inference_time * 1000,
                    **metrics,
                    **{f'loss_{k}': v for k, v in loss_components.items()}
                }

                results.append(result)

            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
                continue

        # Hide progress bar
        self.batch_progress.setVisible(False)

        if not results:
            QMessageBox.warning(self, "Warning", "No results were generated.")
            return

        # Create output directory
        output_dir = os.path.join(config.results_path, f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(output_dir, exist_ok=True)

        # Save results to CSV
        results_df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "batch_metrics.csv")
        results_df.to_csv(csv_path, index=False)

        # Generate summary plots
        self._generate_batch_summary_plots(results_df, output_dir)

        QMessageBox.information(
            self, "Batch Processing Complete",
            f"Processed {len(results)} files.\nResults saved to {output_dir}"
        )
    def _generate_batch_summary_plots(self, results_df, output_dir):
        """Generate summary plots for batch processing results."""
        # Create figure for key metrics
        plt.figure(figsize=(15, 10))

        # Plot PSNR
        plt.subplot(221)
        plt.hist(results_df['psnr'], bins=20, color='green', alpha=0.7)
        plt.axvline(results_df['psnr'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["psnr"].mean():.2f} dB')
        plt.title('PSNR Distribution')
        plt.xlabel('PSNR (dB)')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot SSIM
        plt.subplot(222)
        plt.hist(results_df['ssim'], bins=20, color='blue', alpha=0.7)
        plt.axvline(results_df['ssim'].mean(), color='r', linestyle='--',
                   label=f'Mean: {results_df["ssim"].mean():.4f}')
        plt.title('SSIM Distribution')
        plt.xlabel('SSIM')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot RMSE
        plt.subplot(223)
        plt.hist(results_df['rmse'], bins=20, color='red', alpha=0.7)
        plt.axvline(results_df['rmse'].mean(), color='k', linestyle='--',
                   label=f'Mean: {results_df["rmse"].mean():.4f}')
        plt.title('RMSE Distribution')
        plt.xlabel('RMSE')
        plt.ylabel('Frequency')
        plt.legend()

        # Plot inference time
        plt.subplot(224)
        plt.hist(results_df['inference_time_ms'], bins=20, color='orange', alpha=0.7)
        plt.axvline(results_df['inference_time_ms'].mean(), color='k', linestyle='--',
                   label=f'Mean: {results_df["inference_time_ms"].mean():.2f} ms')
        plt.title('Inference Time Distribution')
        plt.xlabel('Time (ms)')
        plt.ylabel('Frequency')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "batch_metrics_summary.png"), dpi=300)
        plt.close()

        # Create additional plot for spectral smoothness metrics if available
        if 'loss_spectral_smoothness_loss' in results_df.columns:
            plt.figure(figsize=(10, 8))
            plt.hist(results_df['loss_spectral_smoothness_loss'], bins=20, color='purple', alpha=0.7)
            plt.axvline(results_df['loss_spectral_smoothness_loss'].mean(), color='k', linestyle='--',
                       label=f'Mean: {results_df["loss_spectral_smoothness_loss"].mean():.6f}')
            plt.title('Spectral Smoothness Loss Distribution')
            plt.xlabel('Smoothness Loss')
            plt.ylabel('Frequency')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "spectral_smoothness_summary.png"), dpi=300)
            plt.close()

        # Create summary table
        summary = {
            'Metric': ['PSNR', 'SSIM', 'RMSE', 'MRAE', 'Spectral Fidelity', 'Inference Time'],
            'Mean': [
                results_df['psnr'].mean(),
                results_df['ssim'].mean(),
                results_df['rmse'].mean(),
                results_df['mrae'].mean(),
                results_df['spectral_fidelity'].mean(),
                results_df['inference_time_ms'].mean()
            ],
            'Std': [
                results_df['psnr'].std(),
                results_df['ssim'].std(),
                results_df['rmse'].std(),
                results_df['mrae'].std(),
                results_df['spectral_fidelity'].std(),
                results_df['inference_time_ms'].std()
            ],
            'Min': [
                results_df['psnr'].min(),
                results_df['ssim'].min(),
                results_df['rmse'].min(),
                results_df['mrae'].min(),
                results_df['spectral_fidelity'].min(),
                results_df['inference_time_ms'].min()
            ],
            'Max': [
                results_df['psnr'].max(),
                results_df['ssim'].max(),
                results_df['rmse'].max(),
                results_df['mrae'].max(),
                results_df['spectral_fidelity'].max(),
                results_df['inference_time_ms'].max()
            ]
        }

        # Add smoothness metrics if available
        if 'loss_spectral_smoothness_loss' in results_df.columns:
            summary['Metric'].append('Spectral Smoothness Loss')
            summary['Mean'].append(results_df['loss_spectral_smoothness_loss'].mean())
            summary['Std'].append(results_df['loss_spectral_smoothness_loss'].std())
            summary['Min'].append(results_df['loss_spectral_smoothness_loss'].min())
            summary['Max'].append(results_df['loss_spectral_smoothness_loss'].max())

        if 'loss_spectral_tv_loss' in results_df.columns:
            summary['Metric'].append('Spectral TV Loss')
            summary['Mean'].append(results_df['loss_spectral_tv_loss'].mean())
            summary['Std'].append(results_df['loss_spectral_tv_loss'].std())
            summary['Min'].append(results_df['loss_spectral_tv_loss'].min())
            summary['Max'].append(results_df['loss_spectral_tv_loss'].max())

        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(output_dir, "batch_summary.csv"), index=False)


def main():
    app = QApplication(sys.argv)
    viewer = ReconstructionViewer()
    viewer.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()import sys
