import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import math
import os
from config import output_dir, ADAPTIVE_ATTENTION
from epar.position_attention_model import PositionAttentionConfig, PositionEffectFunction

# Set font support for English
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class AdaptiveAttentionConfig:
    """Adaptive Attention Configuration Parameters"""
    sequence_length: int = 512
    hidden_dim: int = 768
    num_heads: int = 12
    alpha: float = 1.0
    beta: float = 2.0
    temperature: float = 1.0
    
    # Adaptive parameters
    task_aware: bool = True
    content_aware: bool = True
    adaptive_learning_rate: float = 0.01
    task_embedding_dim: int = 64
    content_embedding_dim: int = 64

class TaskAwareModule(nn.Module):
    """Task-Aware Module"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        super().__init__()
        self.config = config
        
        # Task embedding
        self.task_embedding = nn.Embedding(10, config.task_embedding_dim)  # Support 10 task types
        
        # Task weight generator
        self.task_weight_generator = nn.Sequential(
            nn.Linear(config.task_embedding_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Sigmoid()
        )
        
        # Task type classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 10),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, task_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            task_id: Task ID [batch_size] or None (auto-inference)
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Task weights and task type
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        if task_id is None:
            # Auto-inference task type
            # Use sequence average representation to infer task
            sequence_repr = x.mean(dim=1)  # [batch_size, hidden_dim]
            task_probs = self.task_classifier(sequence_repr)
            task_id = torch.argmax(task_probs, dim=-1)
        
        # Get task embedding
        task_emb = self.task_embedding(task_id)  # [batch_size, task_embedding_dim]
        
        # Generate task weights
        task_weights = self.task_weight_generator(task_emb)  # [batch_size, hidden_dim]
        task_weights = task_weights.unsqueeze(1).expand(-1, seq_len, -1)  # [batch_size, seq_len, hidden_dim]
        
        return task_weights, task_id

class ContentAwareModule(nn.Module):
    """Content-Aware Module"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        super().__init__()
        self.config = config
        
        # Content importance estimator
        self.content_importance_estimator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Content type classifier
        self.content_type_classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 5),  # 5 content types
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward propagation
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Content importance weights and content types
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Calculate content importance for each position
        content_importance = self.content_importance_estimator(x)  # [batch_size, seq_len, 1]
        content_importance = content_importance.squeeze(-1)  # [batch_size, seq_len]
        
        # Classify content types
        content_types = self.content_type_classifier(x)  # [batch_size, seq_len, 5]
        
        return content_importance, content_types

class AdaptivePositionAwareAttention(nn.Module):
    """Adaptive Position-Aware Attention Mechanism"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        super().__init__()
        self.config = config
        self.position_effect = PositionEffectFunction(config.alpha, config.beta)
        
        # Basic attention layers
        self.query_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.output_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, config.sequence_length, config.hidden_dim)
        )
        
        # Adaptive modules
        if config.task_aware:
            self.task_aware_module = TaskAwareModule(config)
        
        if config.content_aware:
            self.content_aware_module = ContentAwareModule(config)
        
        # Adaptive weight fusion
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(3, config.hidden_dim),  # Average of 3 attention matrices
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),  # Output single fusion weight value
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                task_id: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Forward propagation
        
        Args:
            x: Input sequence [batch_size, seq_len, hidden_dim]
            mask: Attention mask
            task_id: Task ID
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: Output, attention weights and adaptive information
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Add position embedding
        x = x + self.position_embedding[:, :seq_len, :]
        
        # Calculate basic Q, K, V
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)
        
        # Calculate basic attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hidden_dim)
        
        # Apply position effect function
        position_weights = self.position_effect.get_position_matrix(seq_len).to(x.device)
        position_weights = position_weights.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Basic attention weights
        base_attention = attention_scores * position_weights
        
        # Adaptive attention mechanism
        adaptive_info = {}
        
        if self.config.task_aware:
            # Task-aware attention
            task_weights, detected_task = self.task_aware_module(x, task_id)
            # Expand task weights to attention matrix dimensions [batch_size, seq_len, hidden_dim] -> [batch_size, seq_len, seq_len]
            task_weights_expanded = task_weights.mean(dim=-1, keepdim=True).expand(-1, -1, seq_len)
            task_attention = base_attention * task_weights_expanded
            adaptive_info['task_weights'] = task_weights
            adaptive_info['detected_task'] = detected_task
            adaptive_info['task_attention'] = task_attention
        else:
            task_attention = base_attention
        
        if self.config.content_aware:
            # Content-aware attention
            content_importance, content_types = self.content_aware_module(x)
            # Expand content importance to attention matrix dimensions [batch_size, seq_len] -> [batch_size, seq_len, seq_len]
            content_importance_expanded = content_importance.unsqueeze(-1).expand(-1, -1, seq_len)
            content_attention = base_attention * content_importance_expanded
            adaptive_info['content_importance'] = content_importance
            adaptive_info['content_types'] = content_types
            adaptive_info['content_attention'] = content_attention
        else:
            content_attention = base_attention
        
        # Fuse adaptive attention
        if self.config.task_aware and self.config.content_aware:
            # Fuse task-aware and content-aware attention
            fusion_input = torch.cat([
                base_attention.mean(dim=-1, keepdim=True),
                task_attention.mean(dim=-1, keepdim=True),
                content_attention.mean(dim=-1, keepdim=True)
            ], dim=-1)
            
            fusion_weights = self.adaptive_fusion(fusion_input)  # [batch_size, seq_len, 1]
            
            # Apply fusion weights - ensure dimension matching
            final_attention = (base_attention * (1 - fusion_weights) + 
                             task_attention * fusion_weights * 0.5 +
                             content_attention * fusion_weights * 0.5)
        else:
            final_attention = base_attention
        
        # Apply mask
        if mask is not None:
            final_attention = final_attention.masked_fill(mask == 0, float('-inf'))
        
        # Apply temperature parameter
        final_attention = final_attention / self.config.temperature
        
        # Calculate attention weights
        attention_weights = F.softmax(final_attention, dim=-1)
        
        # Calculate output
        output = torch.matmul(attention_weights, V)
        output = self.output_proj(output)
        
        return output, attention_weights, adaptive_info

class AdaptiveAttentionExperiment:
    """Adaptive Attention Experiment Class"""
    
    def __init__(self, config: AdaptiveAttentionConfig):
        self.config = config
        self.model = AdaptivePositionAwareAttention(config)
        self.results = {}
    
    def generate_synthetic_data(self, batch_size: int = 4, 
                               task_types: Optional[List[int]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate synthetic data"""
        if task_types is None:
            task_types = [np.random.randint(0, 10) for _ in range(batch_size)]
        
        # Generate random sequences
        data = torch.randn(batch_size, self.config.sequence_length, self.config.hidden_dim)
        task_ids = torch.tensor(task_types, dtype=torch.long)
        
        return data, task_ids
    
    def analyze_adaptive_attention(self, data: torch.Tensor, 
                                  task_ids: torch.Tensor) -> Dict:
        """Analyze adaptive attention mechanism"""
        self.model.eval()
        
        with torch.no_grad():
            # Forward propagation
            output, attention_weights, adaptive_info = self.model(data, task_id=task_ids)
            
            # Analysis results
            analysis = {
                'attention_weights': attention_weights,
                'output': output,
                'adaptive_info': adaptive_info,
                'task_ids': task_ids
            }
            
            # Calculate adaptive metrics
            if self.config.task_aware:
                analysis['task_metrics'] = self._calculate_task_metrics(adaptive_info)
            
            if self.config.content_aware:
                analysis['content_metrics'] = self._calculate_content_metrics(adaptive_info)
            
            # Calculate attention distribution
            analysis['attention_distribution'] = self._calculate_attention_distribution(attention_weights)
            
            return analysis
    
    def _calculate_task_metrics(self, adaptive_info: Dict) -> Dict:
        """Calculate task-related metrics"""
        task_weights = adaptive_info['task_weights']
        detected_task = adaptive_info['detected_task']
        
        return {
            'task_weight_mean': task_weights.mean().item(),
            'task_weight_std': task_weights.std().item(),
            'task_weight_range': (task_weights.min().item(), task_weights.max().item()),
            'detected_task_accuracy': (detected_task == adaptive_info.get('task_ids', detected_task)).float().mean().item()
        }
    
    def _calculate_content_metrics(self, adaptive_info: Dict) -> Dict:
        """Calculate content-related metrics"""
        content_importance = adaptive_info['content_importance']
        content_types = adaptive_info['content_types']
        
        return {
            'content_importance_mean': content_importance.mean().item(),
            'content_importance_std': content_importance.std().item(),
            'content_importance_range': (content_importance.min().item(), content_importance.max().item()),
            'content_type_diversity': content_types.argmax(dim=-1).unique().shape[0]
        }
    
    def _calculate_attention_distribution(self, attention_weights: torch.Tensor) -> Dict:
        """Calculate attention distribution"""
        return {
            'mean_attention': attention_weights.mean(dim=0),
            'std_attention': attention_weights.std(dim=0),
            'max_attention': attention_weights.max(dim=0)[0],
            'min_attention': attention_weights.min(dim=0)[0],
            'attention_entropy': -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
        }
    
    def visualize_adaptive_attention(self, analysis: Dict, save_path: str = None):
        """Visualize adaptive attention results and save to files"""
        if save_path is None:
            save_path = output_dir(ADAPTIVE_ATTENTION)
        os.makedirs(save_path, exist_ok=True)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Adaptive Attention Mechanism Analysis', fontsize=16, fontweight='bold')
        
        # 1. Attention weights heatmap
        attention_weights = analysis['attention_weights'][0].cpu().numpy()
        sns.heatmap(attention_weights, ax=axes[0, 0], cmap='viridis')
        axes[0, 0].set_title('Attention Weights Distribution')
        axes[0, 0].set_xlabel('Key Position')
        axes[0, 0].set_ylabel('Query Position')
        
        # 2. Task weights distribution
        if self.config.task_aware:
            task_weights = analysis['adaptive_info']['task_weights'][0].cpu().numpy()
            axes[0, 1].plot(task_weights)
            axes[0, 1].set_title('Task Weights Distribution')
            axes[0, 1].set_xlabel('Position')
            axes[0, 1].set_ylabel('Task Weight')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Content importance distribution
        if self.config.content_aware:
            content_importance = analysis['adaptive_info']['content_importance'][0].cpu().numpy()
            axes[0, 2].plot(content_importance)
            axes[0, 2].set_title('Content Importance Distribution')
            axes[0, 2].set_xlabel('Position')
            axes[0, 2].set_ylabel('Content Importance')
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Attention entropy distribution
        attention_entropy = analysis['attention_distribution']['attention_entropy'].cpu().numpy()
        axes[1, 0].hist(attention_entropy.flatten(), bins=20, alpha=0.7)
        axes[1, 0].set_title('Attention Entropy Distribution')
        axes[1, 0].set_xlabel('Attention Entropy')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Task type distribution
        if self.config.task_aware:
            detected_tasks = analysis['adaptive_info']['detected_task'].cpu().numpy()
            task_counts = np.bincount(detected_tasks, minlength=10)
            axes[1, 1].bar(range(10), task_counts)
            axes[1, 1].set_title('Detected Task Type Distribution')
            axes[1, 1].set_xlabel('Task Type')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Content type distribution
        if self.config.content_aware:
            content_types = analysis['adaptive_info']['content_types'][0].argmax(dim=-1).cpu().numpy()
            content_counts = np.bincount(content_types, minlength=5)
            axes[1, 2].bar(range(5), content_counts)
            axes[1, 2].set_title('Content Type Distribution')
            axes[1, 2].set_xlabel('Content Type')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save comprehensive visualization
        plt.savefig(os.path.join(save_path, 'adaptive_attention_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Comprehensive visualization saved to: {os.path.join(save_path, 'adaptive_attention_analysis.png')}")
        
        # Close figure to free memory
        plt.close()
        
        # Generate and save detailed charts
        self._save_detailed_visualizations(analysis, save_path)
    
    def _save_detailed_visualizations(self, analysis: Dict, save_path: str):
        """Save detailed individual visualization charts"""
        # 1. Attention weights heatmap
        plt.figure(figsize=(10, 8))
        attention_weights = analysis['attention_weights'][0].cpu().numpy()
        sns.heatmap(attention_weights, cmap='viridis', annot=False)
        plt.title('Attention Weights Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'attention_weights_heatmap.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention weights heatmap saved to: {os.path.join(save_path, 'attention_weights_heatmap.png')}")
        
        # 2. Task weights distribution
        if self.config.task_aware:
            plt.figure(figsize=(10, 6))
            task_weights = analysis['adaptive_info']['task_weights'][0].cpu().numpy()
            plt.plot(task_weights, linewidth=2, color='blue', alpha=0.8)
            plt.title('Task Weights Distribution Across Positions', fontsize=14, fontweight='bold')
            plt.xlabel('Position')
            plt.ylabel('Task Weight')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'task_weights_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Task weights distribution chart saved to: {os.path.join(save_path, 'task_weights_distribution.png')}")
        
        # 3. Content importance distribution
        if self.config.content_aware:
            plt.figure(figsize=(10, 6))
            content_importance = analysis['adaptive_info']['content_importance'][0].cpu().numpy()
            plt.plot(content_importance, linewidth=2, color='green', alpha=0.8)
            plt.title('Content Importance Distribution Across Positions', fontsize=14, fontweight='bold')
            plt.xlabel('Position')
            plt.ylabel('Content Importance')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'content_importance_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Content importance distribution chart saved to: {os.path.join(save_path, 'content_importance_distribution.png')}")
        
        # 4. Attention entropy distribution
        plt.figure(figsize=(10, 6))
        attention_entropy = analysis['attention_distribution']['attention_entropy'].cpu().numpy()
        plt.hist(attention_entropy.flatten(), bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.title('Attention Entropy Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Attention Entropy')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'attention_entropy_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention entropy distribution chart saved to: {os.path.join(save_path, 'attention_entropy_distribution.png')}")
        
        # 5. Task type distribution
        if self.config.task_aware:
            plt.figure(figsize=(10, 6))
            detected_tasks = analysis['adaptive_info']['detected_task'].cpu().numpy()
            task_counts = np.bincount(detected_tasks, minlength=10)
            colors = plt.cm.Set3(np.linspace(0, 1, 10))
            bars = plt.bar(range(10), task_counts, color=colors, alpha=0.8)
            plt.title('Detected Task Type Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Task Type')
            plt.ylabel('Count')
            plt.xticks(range(10))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'task_type_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Task type distribution chart saved to: {os.path.join(save_path, 'task_type_distribution.png')}")
        
        # 6. Content type distribution
        if self.config.content_aware:
            plt.figure(figsize=(10, 6))
            content_types = analysis['adaptive_info']['content_types'][0].argmax(dim=-1).cpu().numpy()
            content_counts = np.bincount(content_types, minlength=5)
            colors = plt.cm.Pastel1(np.linspace(0, 1, 5))
            plt.bar(range(5), content_counts, color=colors, alpha=0.8)
            plt.title('Content Type Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Content Type')
            plt.ylabel('Count')
            plt.xticks(range(5))
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'content_type_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Content type distribution chart saved to: {os.path.join(save_path, 'content_type_distribution.png')}")
        
        # 7. Position attention analysis
        self._save_position_attention_analysis(analysis, save_path)
    
    def _save_position_attention_analysis(self, analysis: Dict, save_path: str):
        """Save position attention analysis charts"""
        # Calculate position attention statistics
        attention_weights = analysis['attention_weights'][0].cpu().numpy()
        position_attention = attention_weights.mean(axis=0)  # Average attention for each position
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Subplot 1: Position attention distribution
        ax1.plot(position_attention, linewidth=2, color='red', alpha=0.8)
        ax1.set_title('Position Attention Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Position')
        ax1.set_ylabel('Average Attention Weight')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Position attention heatmap
        im = ax2.imshow(attention_weights, cmap='viridis', aspect='auto')
        ax2.set_title('Position Attention Heatmap', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Key Position')
        ax2.set_ylabel('Query Position')
        plt.colorbar(im, ax=ax2)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'position_attention_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Position attention analysis chart saved to: {os.path.join(save_path, 'position_attention_analysis.png')}")
    
    def generate_parameter_sensitivity_plots(self, save_path: str = None):
        """Generate parameter sensitivity analysis plots"""
        if save_path is None:
            save_path = output_dir(ADAPTIVE_ATTENTION)
        os.makedirs(save_path, exist_ok=True)
        
        # Test different alpha and beta values
        alpha_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        beta_values = [1.0, 1.5, 2.0, 2.5, 3.0]
        
        # Store results
        results = {}
        
        for alpha in alpha_values:
            for beta in beta_values:
                # Create temporary configuration
                temp_config = AdaptiveAttentionConfig(
                    sequence_length=32,  # Use smaller sequence length to speed up computation
                    hidden_dim=64,
                    num_heads=2,
                    alpha=alpha,
                    beta=beta,
                    task_aware=True,
                    content_aware=True
                )
                
                # Create temporary model
                temp_model = AdaptivePositionAwareAttention(temp_config)
                
                # Generate test data
                test_data = torch.randn(2, 32, 64)
                test_task_ids = torch.randint(0, 10, (2,))
                
                # Forward propagation
                with torch.no_grad():
                    output, attention_weights, adaptive_info = temp_model(test_data, task_id=test_task_ids)
                    
                    # Calculate attention entropy
                    attention_entropy = -(attention_weights * torch.log(attention_weights + 1e-8)).sum(dim=-1).mean()
                    
                    # Calculate statistics for task weights and content importance
                    task_weight_mean = adaptive_info['task_weights'].mean().item()
                    content_importance_mean = adaptive_info['content_importance'].mean().item()
                    
                    results[(alpha, beta)] = {
                        'attention_entropy': attention_entropy.item(),
                        'task_weight_mean': task_weight_mean,
                        'content_importance_mean': content_importance_mean
                    }
        
        # Generate parameter sensitivity plots
        self._plot_parameter_sensitivity(results, save_path)
    
    def _plot_parameter_sensitivity(self, results: Dict, save_path: str):
        """Plot parameter sensitivity charts"""
        # Extract data
        alphas = sorted(list(set([k[0] for k in results.keys()])))
        betas = sorted(list(set([k[1] for k in results.keys()])))
        
        # Create data matrices
        entropy_matrix = np.zeros((len(alphas), len(betas)))
        task_weight_matrix = np.zeros((len(alphas), len(betas)))
        content_importance_matrix = np.zeros((len(alphas), len(betas)))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                if (alpha, beta) in results:
                    entropy_matrix[i, j] = results[(alpha, beta)]['attention_entropy']
                    task_weight_matrix[i, j] = results[(alpha, beta)]['task_weight_mean']
                    content_importance_matrix[i, j] = results[(alpha, beta)]['content_importance_mean']
        
        # 1. Attention entropy sensitivity to alpha-beta parameters
        plt.figure(figsize=(10, 8))
        sns.heatmap(entropy_matrix, 
                    xticklabels=betas, 
                    yticklabels=alphas,
                    annot=True, 
                    fmt='.3f',
                    cmap='viridis')
        plt.title('Attention Entropy Sensitivity to Alpha and Beta Parameters', fontsize=14, fontweight='bold')
        plt.xlabel('Beta Parameter')
        plt.ylabel('Alpha Parameter')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'parameter_sensitivity_entropy.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter sensitivity entropy chart saved to: {os.path.join(save_path, 'parameter_sensitivity_entropy.png')}")
        
        # 2. Task weight sensitivity to alpha-beta parameters
        plt.figure(figsize=(10, 8))
        sns.heatmap(task_weight_matrix, 
                    xticklabels=betas, 
                    yticklabels=alphas,
                    annot=True, 
                    fmt='.3f',
                    cmap='Blues')
        plt.title('Task Weight Sensitivity to Alpha and Beta Parameters', fontsize=14, fontweight='bold')
        plt.xlabel('Beta Parameter')
        plt.ylabel('Alpha Parameter')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'parameter_sensitivity_task_weights.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter sensitivity task weights chart saved to: {os.path.join(save_path, 'parameter_sensitivity_task_weights.png')}")
        
        # 3. Content importance sensitivity to alpha-beta parameters
        plt.figure(figsize=(10, 8))
        sns.heatmap(content_importance_matrix, 
                    xticklabels=betas, 
                    yticklabels=alphas,
                    annot=True, 
                    fmt='.3f',
                    cmap='Greens')
        plt.title('Content Importance Sensitivity to Alpha and Beta Parameters', fontsize=14, fontweight='bold')
        plt.xlabel('Beta Parameter')
        plt.ylabel('Alpha Parameter')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'parameter_sensitivity_content_importance.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Parameter sensitivity content importance chart saved to: {os.path.join(save_path, 'parameter_sensitivity_content_importance.png')}")
    
    def run_experiment(self, num_experiments: int = 5) -> Dict:
        """Run complete experiment"""
        print("Starting adaptive attention mechanism experiment...")
        
        all_results = []
        
        for exp_idx in range(num_experiments):
            print(f"Experiment {exp_idx + 1}/{num_experiments}")
            
            # Generate data
            data, task_ids = self.generate_synthetic_data()
            
            # Analyze adaptive attention
            results = self.analyze_adaptive_attention(data, task_ids)
            all_results.append(results)
            
            print(f"  - Task ID: {task_ids.tolist()}")
            if self.config.task_aware:
                print(f"  - Task Weight Mean: {results['task_metrics']['task_weight_mean']:.4f}")
            if self.config.content_aware:
                print(f"  - Content Importance Mean: {results['content_metrics']['content_importance_mean']:.4f}")
        
        # Summarize results
        summary = self._summarize_results(all_results)
        
        # Visualize results of the last experiment
        self.visualize_adaptive_attention(all_results[-1])
        
        return summary
    
    def _summarize_results(self, all_results: List[Dict]) -> Dict:
        """Summarize experimental results"""
        summary = {
            'experiment_count': len(all_results),
            'attention_entropy': [],
            'task_metrics': [],
            'content_metrics': []
        }
        
        for result in all_results:
            summary['attention_entropy'].append(
                result['attention_distribution']['attention_entropy'].item()
            )
            
            if self.config.task_aware:
                summary['task_metrics'].append(result['task_metrics'])
            
            if self.config.content_aware:
                summary['content_metrics'].append(result['content_metrics'])
        
        # Calculate statistics
        summary['attention_entropy'] = {
            'mean': np.mean(summary['attention_entropy']),
            'std': np.std(summary['attention_entropy']),
            'min': np.min(summary['attention_entropy']),
            'max': np.max(summary['attention_entropy'])
        }
        
        return summary

def main():
    """Main function: Run adaptive attention experiment"""
    print("=== Adaptive Attention Mechanism Research Experiment ===")
    
    # Configuration parameters
    config = AdaptiveAttentionConfig(
        sequence_length=64,
        hidden_dim=128,
        num_heads=4,
        alpha=1.0,
        beta=2.0,
        temperature=1.0,
        task_aware=True,
        content_aware=True,
        adaptive_learning_rate=0.01
    )
    
    # Create experiment
    experiment = AdaptiveAttentionExperiment(config)
    
    # Run experiment
    results = experiment.run_experiment(num_experiments=3)
    
    # Print result summary
    print("\n=== Experimental Result Summary ===")
    print(f"Number of Experiments: {results['experiment_count']}")
    print(f"Attention Entropy Statistics:")
    print(f"  - Mean: {results['attention_entropy']['mean']:.4f}")
    print(f"  - Standard Deviation: {results['attention_entropy']['std']:.4f}")
    print(f"  - Range: [{results['attention_entropy']['min']:.4f}, {results['attention_entropy']['max']:.4f}]")
    
    print("\n=== Adaptive Mechanism Verification ===")
    print("1. Task-Aware Attention: Attention_task = Attention_base * Task_Weight(task)")
    print("2. Content-Aware Attention: Attention_content = Attention_base * Content_Importance(content)")
    print("3. Adaptive Fusion: Dynamically balance task-aware and content-aware attention")
    
    # Generate parameter sensitivity analysis plots
    print("\n=== Generating Parameter Sensitivity Analysis ===")
    experiment.generate_parameter_sensitivity_plots()

    print("\n=== All visualizations have been saved to 'output/adaptive' directory ===")
    
    return results

if __name__ == "__main__":
    main()
