"""Position-Aware Attention model and experiment (core module)."""
import sys
import os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import math

from config import output_dir, POSITION_EFFECT

# Set font support for English
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Unified output directory
_OUT_DIR = output_dir(POSITION_EFFECT)

@dataclass
class PositionAttentionConfig:
    """Position Attention Configuration Parameters"""
    sequence_length: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    alpha: float = 1.0  # Position influence strength parameter
    beta: float = 2.0   # Position decay parameter
    gamma: float = 1.5  # Position enhancement parameter (new)
    temperature: float = 1.0  # Attention temperature parameter

class PositionEffectFunction:
    """Position Effect Function Class"""

    def __init__(self, alpha: float = 1.0, beta: float = 2.0, gamma: float = 1.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, i: int, j: int, L: int) -> float:
        """
        Calculate enhanced position effect function

        Args:
            i: Query position
            j: Key position
            L: Sequence length

        Returns:
            float: Enhanced position influence weight
        """
        # Enhanced mathematical modeling: improved position effect function
        # P_effect = α * (1 + γ * e^(-β * |i-j|/L)) / (1 + γ)
        distance = abs(i - j)
        normalized_distance = distance / L

        base_effect = math.exp(-self.beta * normalized_distance)
        enhanced_effect = (1 + self.gamma * base_effect) / (1 + self.gamma)

        return self.alpha * enhanced_effect

    def get_position_matrix(self, L: int) -> torch.Tensor:
        """Get complete position effect matrix"""
        matrix = torch.zeros(L, L)
        for i in range(L):
            for j in range(L):
                matrix[i, j] = self(i, j, L)
        return matrix

class PositionAwareAttention(nn.Module):
    """Position-Aware Attention Mechanism"""

    def __init__(self, config: PositionAttentionConfig):
        super().__init__()
        self.config = config
        self.position_effect = PositionEffectFunction(config.alpha, config.beta)

        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)

        self.position_embedding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.hidden_dim)
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = x.shape

        x = x + self.position_embedding[:, :seq_len, :]

        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)

        position_weights = self.position_effect.get_position_matrix(seq_len).to(x.device)
        position_weights = position_weights.unsqueeze(0).expand(batch_size, -1, -1)

        attention_scores = attention_scores * position_weights

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        attention_scores = attention_scores / self.config.temperature

        attention_weights = F.softmax(attention_scores, dim=-1)

        output = torch.matmul(attention_weights, V)
        output = self.output_proj(output)

        return output, attention_weights

class PositionAttentionExperiment:
    """Position Attention Experiment Class"""

    def __init__(self, config: PositionAttentionConfig):
        self.config = config
        self.model = PositionAwareAttention(config)
        self.results = {}

    def generate_synthetic_data(self, batch_size: int = 4) -> torch.Tensor:
        data = torch.randn(batch_size, self.config.sequence_length, self.config.hidden_dim)
        return data

    def analyze_position_effects(self, data: torch.Tensor) -> Dict:
        self.model.eval()

        with torch.no_grad():
            output, attention_weights = self.model(data)

            batch_size, seq_len, _ = attention_weights.shape

            position_stats = {
                'mean_attention': attention_weights.mean(dim=0),
                'std_attention': attention_weights.std(dim=0),
                'max_attention': attention_weights.max(dim=0)[0],
                'min_attention': attention_weights.min(dim=0)[0]
            }

            position_correlation = self._calculate_position_correlation(attention_weights)
            optimal_positions = self._find_optimal_positions(attention_weights, data)

            return {
                'attention_weights': attention_weights,
                'position_stats': position_stats,
                'position_correlation': position_correlation,
                'optimal_positions': optimal_positions
            }

    def _calculate_position_correlation(self, attention_weights: torch.Tensor) -> torch.Tensor:
        seq_len = attention_weights.shape[1]
        correlation_matrix = torch.zeros(seq_len, seq_len)

        for i in range(seq_len):
            for j in range(seq_len):
                attention_i = attention_weights[:, i, :].flatten()
                attention_j = attention_weights[:, j, :].flatten()
                correlation = torch.corrcoef(torch.stack([attention_i, attention_j]))[0, 1]
                correlation_matrix[i, j] = correlation if not torch.isnan(correlation) else 0.0

        return correlation_matrix

    def _find_optimal_positions(self, attention_weights: torch.Tensor, data: torch.Tensor) -> Dict:
        seq_len = attention_weights.shape[1]
        information_importance = torch.norm(data, dim=-1)
        attention_info_product = attention_weights * information_importance.unsqueeze(-1)
        position_scores = attention_info_product.sum(dim=-1)
        optimal_positions = torch.argmax(position_scores, dim=-1)

        return {
            'optimal_positions': optimal_positions,
            'position_scores': position_scores,
            'information_importance': information_importance
        }

    def visualize_results(self, results: Dict):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Position-Attention Quantitative Relationship Analysis', fontsize=16, fontweight='bold')

        attention_weights = results['attention_weights'][0].cpu().numpy()
        sns.heatmap(attention_weights, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Attention Weights Distribution')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')

        seq_len = attention_weights.shape[0]
        position_effect = self.model.position_effect.get_position_matrix(seq_len).cpu().numpy()
        sns.heatmap(position_effect, ax=axes[0, 1], cmap='plasma')
        axes[0, 1].set_title('Position Effect Function')
        axes[0, 1].set_xlabel('Key Position')
        axes[0, 1].set_ylabel('Query Position')

        correlation = results['position_correlation'].cpu().numpy()
        sns.heatmap(correlation, ax=axes[0, 2], cmap='RdBu_r', center=0)
        axes[0, 2].set_title('Position Correlation')
        axes[0, 2].set_xlabel('Position j')
        axes[0, 2].set_ylabel('Position i')

        mean_attention = results['position_stats']['mean_attention'].cpu().numpy()
        sns.heatmap(mean_attention, ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Average Attention Distribution')
        axes[1, 0].set_xlabel('Key Position')
        axes[1, 0].set_ylabel('Query Position')

        info_importance = results['optimal_positions']['information_importance'][0].cpu().numpy()
        axes[1, 1].plot(info_importance)
        axes[1, 1].set_title('Information Importance Distribution')
        axes[1, 1].set_xlabel('Position')
        axes[1, 1].set_ylabel('Information Importance')
        axes[1, 1].grid(True, alpha=0.3)

        position_scores = results['optimal_positions']['position_scores'][0].cpu().numpy()
        optimal_pos = results['optimal_positions']['optimal_positions'][0].cpu().numpy()
        axes[1, 2].plot(position_scores)
        axes[1, 2].axvline(x=optimal_pos, color='red', linestyle='--',
                           label=f'Optimal Position: {optimal_pos}')
        axes[1, 2].set_title('Position Score Distribution')
        axes[1, 2].set_xlabel('Position')
        axes[1, 2].set_ylabel('Attention-Information Score')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_attention_analysis.png'), dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {_OUT_DIR}/position_attention_analysis.png")
        plt.close()
        self._save_detailed_visualizations(results)

    def _save_detailed_visualizations(self, results: Dict):
        plt.figure(figsize=(10, 8))
        attention_weights = results['attention_weights'][0].cpu().numpy()
        sns.heatmap(attention_weights, cmap='viridis', annot=False)
        plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'attention_weights_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention weights heatmap saved to: {_OUT_DIR}/attention_weights_heatmap.png")

        plt.figure(figsize=(10, 8))
        seq_len = attention_weights.shape[0]
        position_effect = self.model.position_effect.get_position_matrix(seq_len).cpu().numpy()
        sns.heatmap(position_effect, cmap='plasma', annot=False)
        plt.title('Position Effect Function Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_effect_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position effect matrix saved to: {_OUT_DIR}/position_effect_matrix.png")

        plt.figure(figsize=(10, 8))
        correlation = results['position_correlation'].cpu().numpy()
        sns.heatmap(correlation, cmap='RdBu_r', center=0, annot=False)
        plt.title('Position Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xlabel('Position j')
        plt.ylabel('Position i')
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position correlation matrix saved to: {_OUT_DIR}/position_correlation.png")

        plt.figure(figsize=(10, 8))
        mean_attention = results['position_stats']['mean_attention'].cpu().numpy()
        sns.heatmap(mean_attention, cmap='viridis', annot=False)
        plt.title('Average Attention Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'average_attention_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Average attention distribution saved to: {_OUT_DIR}/average_attention_distribution.png")

        plt.figure(figsize=(10, 6))
        info_importance = results['optimal_positions']['information_importance'][0].cpu().numpy()
        plt.plot(info_importance, linewidth=2, color='blue', alpha=0.8)
        plt.title('Information Importance Distribution Across Positions', fontsize=14, fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Information Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'information_importance_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Information importance distribution saved to: {_OUT_DIR}/information_importance_distribution.png")

        plt.figure(figsize=(10, 6))
        position_scores = results['optimal_positions']['position_scores'][0].cpu().numpy()
        optimal_pos = results['optimal_positions']['optimal_positions'][0].cpu().numpy()
        plt.plot(position_scores, linewidth=2, color='green', alpha=0.8)
        plt.axvline(x=optimal_pos, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal Position: {optimal_pos}')
        plt.title('Position Score Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Position')
        plt.ylabel('Attention-Information Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_score_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position score distribution saved to: {_OUT_DIR}/position_score_distribution.png")

        self._save_position_attention_analysis(results)
        self._save_statistical_analysis(results)

    def _save_position_attention_analysis(self, results: Dict):
        attention_weights = results['attention_weights'][0].cpu().numpy()
        position_attention = attention_weights.mean(axis=0)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        ax1.plot(position_attention, linewidth=2, color='red', alpha=0.8)
        ax1.set_title('Position Attention Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Average Attention Weight')
        ax1.grid(True, alpha=0.3)
        im = ax2.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax2.set_title('Position Attention Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_attention_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position attention analysis chart saved to: {_OUT_DIR}/position_attention_analysis.png")

    def _save_statistical_analysis(self, results: Dict):
        attention_weights = results['attention_weights'][0].cpu().numpy()
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Attention Weights Statistical Analysis', fontsize=16, fontweight='bold')
        mean_attention = attention_weights.mean(axis=0)
        ax1.plot(mean_attention, linewidth=2, color='blue', alpha=0.8)
        ax1.set_title('Mean Attention per Position')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Mean Attention Weight')
        ax1.grid(True, alpha=0.3)
        std_attention = attention_weights.std(axis=0)
        ax2.plot(std_attention, linewidth=2, color='red', alpha=0.8)
        ax2.set_title('Standard Deviation per Position')
        ax2.set_xlabel('Position')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        max_attention = attention_weights.max(axis=0)
        ax3.plot(max_attention, linewidth=2, color='green', alpha=0.8)
        ax3.set_title('Maximum Attention per Position')
        ax3.set_xlabel('Position')
        ax3.set_ylabel('Maximum Attention Weight')
        ax3.grid(True, alpha=0.3)
        min_attention = attention_weights.min(axis=0)
        ax4.plot(min_attention, linewidth=2, color='orange', alpha=0.8)
        ax4.set_title('Minimum Attention per Position')
        ax4.set_xlabel('Position')
        ax4.set_ylabel('Minimum Attention Weight')
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'attention_statistics.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention statistics chart saved to: {_OUT_DIR}/attention_statistics.png")

        seq_len = attention_weights.shape[0]
        position_effect = self.model.position_effect.get_position_matrix(seq_len).cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Position Effect Function Analysis', fontsize=16, fontweight='bold')
        distances = range(0, min(20, seq_len))
        effect_values = [self.model.position_effect(0, d, seq_len) for d in distances]
        ax1.plot(distances, effect_values, 'o-', linewidth=2, markersize=6, color='purple')
        ax1.set_title('Position Effect vs Distance')
        ax1.set_xlabel('Distance from Query Position')
        ax1.set_ylabel('Position Effect Weight')
        ax1.grid(True, alpha=0.3)
        im = ax2.imshow(position_effect, cmap='plasma', aspect='auto')
        ax2.set_title('Position Effect Function Surface')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax2)
        plt.tight_layout()
        plt.savefig(os.path.join(_OUT_DIR, 'position_effect_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position effect analysis saved to: {_OUT_DIR}/position_effect_analysis.png")

    def run_experiment(self, num_experiments: int = 5) -> Dict:
        print("Starting Position-Attention Quantitative Relationship Experiment...")
        all_results = []
        for exp_idx in range(num_experiments):
            print(f"Experiment {exp_idx + 1}/{num_experiments}")
            data = self.generate_synthetic_data()
            results = self.analyze_position_effects(data)
            all_results.append(results)
            print(f"  - Optimal Position: {results['optimal_positions']['optimal_positions'][0].item()}")
            print(f"  - Average Attention Weight: {results['attention_weights'].mean().item():.4f}")
        summary = self._summarize_results(all_results)
        self.visualize_results(all_results[-1])
        return summary

    def _summarize_results(self, all_results: List[Dict]) -> Dict:
        optimal_positions = []
        mean_attention_weights = []
        for result in all_results:
            optimal_positions.append(result['optimal_positions']['optimal_positions'].cpu().numpy())
            mean_attention_weights.append(result['attention_weights'].mean().cpu().numpy())
        optimal_positions = np.concatenate(optimal_positions)
        mean_attention_weights = np.array(mean_attention_weights)
        return {
            'optimal_positions': {
                'mean': float(optimal_positions.mean()),
                'std': float(optimal_positions.std()),
                'min': int(optimal_positions.min()),
                'max': int(optimal_positions.max()),
                'distribution': optimal_positions.tolist()
            },
            'attention_weights': {
                'mean': float(mean_attention_weights.mean()),
                'std': float(mean_attention_weights.std()),
                'min': float(mean_attention_weights.min()),
                'max': float(mean_attention_weights.max())
            },
            'experiment_count': len(all_results)
        }

def main():
    """Main function: Run position-attention experiment"""
    print("=== Position-Attention Quantitative Relationship Research Experiment ===")
    config = PositionAttentionConfig(
        sequence_length=128,
        hidden_dim=256,
        num_heads=8,
        alpha=1.0,
        beta=2.0,
        gamma=1.5,
        temperature=1.0
    )
    experiment = PositionAttentionExperiment(config)
    results = experiment.run_experiment(num_experiments=3)
    print("\n=== Experimental Result Summary ===")
    print(f"Number of Experiments: {results['experiment_count']}")
    print(f"Optimal Position Statistics:")
    print(f"  - Mean: {results['optimal_positions']['mean']:.2f}")
    print(f"  - Standard Deviation: {results['optimal_positions']['std']:.2f}")
    print(f"  - Range: [{results['optimal_positions']['min']}, {results['optimal_positions']['max']}]")
    print(f"Attention Weight Statistics:")
    print(f"  - Mean: {results['attention_weights']['mean']:.4f}")
    print(f"  - Standard Deviation: {results['attention_weights']['std']:.4f}")
    print("\n=== Core Mathematical Modeling Verification ===")
    print("1. Position Effect Function: P_effect = α * (1 + γ * e^(-β * |i-j|/L)) / (1 + γ)")
    print("2. Attention Weights: A_ij = (Q_i^T K_j / √d_k) * P_effect(i,j,L)")
    print("3. Optimal Position: pos* = argmax_i Σ_j A_ij * I_j")
    return results

if __name__ == "__main__":
    main()
