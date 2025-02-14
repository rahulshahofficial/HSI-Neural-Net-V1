import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import seaborn as sns
from tqdm import tqdm
import os

class FilterArrangementEvaluator:
    def __init__(self, hyperspectral_images, model_class, num_arrangements=10, base_seed=42):
        """
        Args:
            hyperspectral_images: Training images
            model_class: Neural network model class to use
            num_arrangements: Number of random arrangements to try
            base_seed: Base seed for reproducibility
        """
        self.images = hyperspectral_images
        self.model_class = model_class
        self.num_arrangements = num_arrangements
        self.base_seed = base_seed
        self.results = []
        self.best_model = None

        # Set random seed for reproducibility
        torch.manual_seed(self.base_seed)
        np.random.seed(self.base_seed)

    def evaluate_arrangement(self, seed):
        """Evaluate a single filter arrangement"""
        print(f"\nEvaluating arrangement with seed {seed}")

        # Create dataset with this seed
        dataset = HyperspectralDataset(self.images, seed=seed)
        filter_map = dataset.get_filter_arrangement()

        # Split data consistently
        torch.manual_seed(self.base_seed)  # Use same seed for consistent splits
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        # Initialize and train model
        model = self.model_class()
        trainer = Trainer(model, train_loader, val_loader)
        final_val_loss = trainer.train(verbose=False)

        return {
            'seed': seed,
            'filter_map': filter_map,
            'val_loss': final_val_loss,
            'model': model,
            'filter_distribution': torch.bincount(filter_map.flatten(),
                                               minlength=config.num_filters)
        }

    def evaluate_all(self):
        """Evaluate all arrangements"""
        print(f"Evaluating {self.num_arrangements} different filter arrangements...")

        for i in tqdm(range(self.num_arrangements)):
            seed = self.base_seed + i
            result = self.evaluate_arrangement(seed)
            self.results.append(result)

        # Sort results by validation loss
        self.results.sort(key=lambda x: x['val_loss'])

        # Store best model
        self.best_model = self.results[0]['model']

        # Print summary
        print("\nResults summary:")
        for i, result in enumerate(self.results):
            print(f"\nArrangement {i+1}:")
            print(f"Seed: {result['seed']}")
            print(f"Validation Loss: {result['val_loss']:.6f}")

        return self.get_best_result()

    def get_best_result(self):
        """Get the best arrangement and model"""
        if not self.results:
            raise ValueError("No arrangements evaluated yet")
        return {
            'model': self.best_model,
            'filter_map': self.results[0]['filter_map'],
            'seed': self.results[0]['seed'],
            'val_loss': self.results[0]['val_loss']
        }

    def visualize_arrangements(self, save_path='results/filter_arrangements'):
        """Visualize all evaluated arrangements"""
        if not self.results:
            print("No arrangements to visualize")
            return

        os.makedirs(save_path, exist_ok=True)

        # Create figure for all arrangements
        num_cols = min(5, self.num_arrangements)
        num_rows = (self.num_arrangements + num_cols - 1) // num_cols

        plt.figure(figsize=(4*num_cols, 4*num_rows))

        for i, result in enumerate(self.results):
            plt.subplot(num_rows, num_cols, i+1)

            # Create heatmap of filter arrangement
            sns.heatmap(result['filter_map'],
                       cmap='viridis',
                       cbar_kws={'label': 'Filter Index'})

            plt.title(f"Seed: {result['seed']}\nLoss: {result['val_loss']:.6f}")
            plt.xlabel('Width')
            plt.ylabel('Height')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'all_arrangements.png'))
        plt.close()

        # Visualize best arrangement in detail
        best_result = self.results[0]
        plt.figure(figsize=(15, 5))

        # Filter arrangement heatmap
        plt.subplot(121)
        sns.heatmap(best_result['filter_map'],
                   cmap='viridis',
                   cbar_kws={'label': 'Filter Index'})
        plt.title('Best Filter Arrangement')
        plt.xlabel('Width')
        plt.ylabel('Height')

        # Filter distribution bar plot
        plt.subplot(122)
        filter_dist = best_result['filter_distribution'].numpy()
        plt.bar(range(len(filter_dist)), filter_dist)
        plt.title('Filter Distribution')
        plt.xlabel('Filter Index')
        plt.ylabel('Pixel Count')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'best_arrangement_analysis.png'))
        plt.close()

    def save_best_model(self, path='results/best_model.pth'):
        """Save the best model and its filter arrangement"""
        if self.best_model is None:
            print("No model trained yet")
            return

        os.makedirs(os.path.dirname(path), exist_ok=True)
        best_result = self.results[0]

        torch.save({
            'model_state_dict': self.best_model.state_dict(),
            'filter_map': best_result['filter_map'],
            'seed': best_result['seed'],
            'val_loss': best_result['val_loss'],
            'filter_distribution': best_result['filter_distribution']
        }, path)
        print(f"Best model and arrangement saved to {path}")

    @classmethod
    def load_best_model(cls, path='results/best_model.pth', model_class=None):
        """Load the best model and its filter arrangement"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"No saved model found at {path}")

        checkpoint = torch.load(path)

        if model_class is None:
            model_class = HyperspectralNet

        model = model_class()
        model.load_state_dict(checkpoint['model_state_dict'])

        return {
            'model': model,
            'filter_map': checkpoint['filter_map'],
            'seed': checkpoint['seed'],
            'val_loss': checkpoint['val_loss'],
            'filter_distribution': checkpoint['filter_distribution']
        }
