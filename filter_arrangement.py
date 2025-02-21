import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import seaborn as sns
from tqdm import tqdm
import os

from config import config
from dataset import HyperspectralDataset
from network import HyperspectralNet
from train import Trainer

class FilterArrangementEvaluator:
    def __init__(self, hyperspectral_images, num_arrangements=None, base_seed=42):
        self.images = hyperspectral_images
        self.num_arrangements = num_arrangements or config.num_arrangements
        self.base_seed = base_seed
        self.results = []
        self.best_model = None

        torch.manual_seed(self.base_seed)
        np.random.seed(self.base_seed)

    def evaluate_arrangement(self, seed):
        dataset = HyperspectralDataset(self.images, seed=seed)
        filter_map = dataset.get_filter_arrangement()

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        model = HyperspectralNet()
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
        for i in tqdm(range(self.num_arrangements), desc="Evaluating arrangements"):
            seed = self.base_seed + i
            result = self.evaluate_arrangement(seed)
            self.results.append(result)

        self.results.sort(key=lambda x: x['val_loss'])
        self.best_model = self.results[0]['model']

        print("\nResults summary:")
        for i, result in enumerate(self.results):
            print(f"\nArrangement {i+1}:")
            print(f"Seed: {result['seed']}")
            print(f"Validation Loss: {result['val_loss']:.6f}")

        return self.get_best_result()

    def get_best_result(self):
        if not self.results:
            raise ValueError("No arrangements evaluated yet")
        return {
            'model': self.best_model,
            'filter_map': self.results[0]['filter_map'],
            'seed': self.results[0]['seed'],
            'val_loss': self.results[0]['val_loss']
        }

    def visualize_arrangements(self):
        if not self.results:
            return

        os.makedirs(config.arrangements_path, exist_ok=True)

        num_cols = min(5, self.num_arrangements)
        num_rows = (self.num_arrangements + num_cols - 1) // num_cols

        plt.figure(figsize=(4*num_cols, 4*num_rows))
        for i, result in enumerate(self.results):
            plt.subplot(num_rows, num_cols, i+1)
            sns.heatmap(result['filter_map'], cmap='viridis',
                       cbar_kws={'label': 'Filter Index'})
            plt.title(f"Seed: {result['seed']}\nLoss: {result['val_loss']:.6f}")

        plt.tight_layout()
        plt.savefig(os.path.join(config.arrangements_path, 'all_arrangements.png'))
        plt.close()

        # Visualize best arrangement
        best_result = self.results[0]
        plt.figure(figsize=(15, 5))

        plt.subplot(121)
        sns.heatmap(best_result['filter_map'], cmap='viridis',
                   cbar_kws={'label': 'Filter Index'})
        plt.title('Best Filter Arrangement')

        plt.subplot(122)
        filter_dist = best_result['filter_distribution'].numpy()
        plt.bar(range(len(filter_dist)), filter_dist)
        plt.title('Filter Distribution')
        plt.xlabel('Filter Index')
        plt.ylabel('Pixel Count')

        plt.tight_layout()
        plt.savefig(os.path.join(config.arrangements_path, 'best_arrangement_analysis.png'))
        plt.close()

    def save_best_model(self):
        if self.best_model is None:
            return

        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)
        best_result = self.results[0]

        torch.save({
            'model_state_dict': self.best_model.state_dict(),
            'filter_map': best_result['filter_map'],
            'seed': best_result['seed'],
            'val_loss': best_result['val_loss'],
            'filter_distribution': best_result['filter_distribution']
        }, config.model_save_path)
