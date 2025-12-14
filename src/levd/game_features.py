"""Game features prediction tracking and evaluation utilities."""
from typing import List
import numpy as np


class GameFeaturesTracker:
    """
    Track game features predictions accuracy using confusion matrix.
    Similar to Arnold's GameFeaturesConfusionMatrix.
    """
    
    def __init__(self, feature_names: List[str]):
        """
        Initialize tracker for game features.
        
        Args:
            feature_names: List of feature names (e.g., ['enemy', 'health'])
        """
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # Confusion matrix components
        self.tp = {name: 0 for name in feature_names}
        self.fp = {name: 0 for name in feature_names}
        self.tn = {name: 0 for name in feature_names}
        self.fn = {name: 0 for name in feature_names}
        
        self.total_samples = 0
    
    def update(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Update confusion matrix with new predictions.
        
        Args:
            predictions: Binary predictions (batch_size, n_features) or (n_features,)
            targets: Ground truth binary labels (batch_size, n_features) or (n_features,)
        """
        # Handle single sample
        if predictions.ndim == 1:
            predictions = predictions.reshape(1, -1)
            targets = targets.reshape(1, -1)
        
        batch_size = predictions.shape[0]
        self.total_samples += batch_size
        
        for i, name in enumerate(self.feature_names):
            for j in range(batch_size):
                pred = predictions[j, i] > 0.5
                target = targets[j, i] > 0.5
                
                if pred and target:
                    self.tp[name] += 1
                elif pred and not target:
                    self.fp[name] += 1
                elif not pred and target:
                    self.fn[name] += 1
                else:
                    self.tn[name] += 1
    
    def get_metrics(self) -> dict:
        """
        Compute precision, recall, F1 for each feature.
        
        Returns:
            Dictionary with metrics for each feature
        """
        metrics = {}
        
        for name in self.feature_names:
            tp = self.tp[name]
            fp = self.fp[name]
            fn = self.fn[name]
            tn = self.tn[name]
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-8)
            
            metrics[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        
        return metrics
    
    def print_stats(self) -> None:
        """Print statistics about game feature predictions."""
        print('\n' + '='*70)
        print('Game Features Prediction Summary')
        print('='*70)
        print(f'Total samples: {self.total_samples}')
        print()
        
        metrics = self.get_metrics()
        
        for name in self.feature_names:
            m = metrics[name]
            print(f'{name.upper():>10}:')
            print(f'  Precision: {m["precision"]:.3f}  |  Recall: {m["recall"]:.3f}  |  F1: {m["f1"]:.3f}')
            print(f'  TP: {m["tp"]:6d}  |  FP: {m["fp"]:6d}  |  FN: {m["fn"]:6d}  |  TN: {m["tn"]:6d}')
            print()
        
        # Overall accuracy
        total_correct = sum(m['tp'] + m['tn'] for m in metrics.values())
        total_predictions = sum(m['tp'] + m['fp'] + m['tn'] + m['fn'] for m in metrics.values())
        overall_accuracy = total_correct / (total_predictions + 1e-8)
        print(f'Overall Accuracy: {overall_accuracy:.3f}')
        print('='*70 + '\n')
    
    def reset(self) -> None:
        """Reset all counters."""
        for name in self.feature_names:
            self.tp[name] = 0
            self.fp[name] = 0
            self.tn[name] = 0
            self.fn[name] = 0
        self.total_samples = 0
