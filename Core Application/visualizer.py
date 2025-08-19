"""
Visualization Module for Sudoku Solver Performance Analysis

This module provides comprehensive visualization capabilities for comparing
and analyzing Sudoku solver performance using matplotlib. It creates
various types of charts and graphs to help understand algorithm behavior,
performance trends, and comparative analysis.

Features:
- Performance comparison charts
- Time complexity visualizations  
- Statistical distribution plots
- Algorithm efficiency analysis
- Interactive plotting capabilities
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import statistics
from performance import PerformanceAnalyzer, PerformanceMetrics
from board import Board


# Set up matplotlib and seaborn styling
plt.style.use('default')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance."""
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 100
    font_size: int = 12
    title_font_size: int = 16
    save_format: str = 'png'
    color_scheme: str = 'viridis'


class SudokuVisualizationSuite:
    """
    Comprehensive visualization suite for Sudoku solver analysis.
    
    This class provides various plotting methods to visualize solver
    performance, compare algorithms, and analyze trends.
    """
    
    def __init__(self, config: VisualizationConfig = None):
        """
        Initialize the visualization suite.
        
        Args:
            config: Configuration for visualization appearance
        """
        self.config = config or VisualizationConfig()
        
        # Set matplotlib parameters
        plt.rcParams['figure.figsize'] = self.config.figure_size
        plt.rcParams['figure.dpi'] = self.config.dpi
        plt.rcParams['font.size'] = self.config.font_size
        
        # Color schemes
        self.colors = {
            'backtracking': '#FF6B6B',
            'dlx': '#4ECDC4', 
            'dancing_links': '#4ECDC4',
            'heuristic': '#45B7D1',
            'basic': '#96CEB4'
        }
    
    def plot_performance_comparison(self, performance_data: Dict[str, PerformanceMetrics],
                                  title: str = "Solver Performance Comparison",
                                  save_path: str = None) -> plt.Figure:
        """
        Create a comparison chart of solver performance metrics.
        
        Args:
            performance_data: Dictionary mapping solver names to PerformanceMetrics
            title: Chart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.config.figure_size)
        fig.suptitle(title, fontsize=self.config.title_font_size, fontweight='bold')
        
        solvers = list(performance_data.keys())
        colors = [self.colors.get(solver.lower(), '#7F8C8D') for solver in solvers]
        
        # Execution Time Comparison
        times = [metrics.execution_time for metrics in performance_data.values()]
        bars1 = ax1.bar(solvers, times, color=colors, alpha=0.7)
        ax1.set_title('Execution Time Comparison')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, time in zip(bars1, times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time:.4f}s', ha='center', va='bottom')
        
        # Steps Taken Comparison
        steps = [metrics.steps_taken for metrics in performance_data.values()]
        bars2 = ax2.bar(solvers, steps, color=colors, alpha=0.7)
        ax2.set_title('Steps Taken Comparison')
        ax2.set_ylabel('Number of Steps')
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, step in zip(bars2, steps):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{step}', ha='center', va='bottom')
        
        # Efficiency Comparison (Steps per Second)
        efficiency = [metrics.steps_taken / max(metrics.execution_time, 0.001) 
                     for metrics in performance_data.values()]
        bars3 = ax3.bar(solvers, efficiency, color=colors, alpha=0.7)
        ax3.set_title('Algorithm Efficiency')
        ax3.set_ylabel('Steps per Second')
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, eff in zip(bars3, efficiency):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{eff:.0f}', ha='center', va='bottom')
        
        # Memory Usage Comparison (if available)
        memory_data = [metrics.memory_usage for metrics in performance_data.values()]
        if any(mem is not None for mem in memory_data):
            memory_values = [mem if mem is not None else 0 for mem in memory_data]
            bars4 = ax4.bar(solvers, memory_values, color=colors, alpha=0.7)
            ax4.set_title('Memory Usage Comparison')
            ax4.set_ylabel('Memory (bytes)')
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, mem in zip(bars4, memory_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{mem}', ha='center', va='bottom')
        else:
            ax4.text(0.5, 0.5, 'Memory data\nnot available', 
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, alpha=0.5)
            ax4.set_title('Memory Usage')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def plot_benchmark_results(self, benchmark_data: Dict[str, Dict],
                              title: str = "Benchmark Results Across Difficulties",
                              save_path: str = None) -> plt.Figure:
        """
        Plot benchmark results across different difficulty levels.
        
        Args:
            benchmark_data: Dictionary with benchmark results
            title: Chart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=self.config.title_font_size, fontweight='bold')
        
        # Extract data for plotting
        algorithms = set()
        difficulties = set()
        
        for key in benchmark_data.keys():
            if '_' in key:
                algo, diff = key.rsplit('_', 1)
                algorithms.add(algo)
                difficulties.add(diff)
        
        algorithms = sorted(list(algorithms))
        difficulties = sorted(list(difficulties))
        
        # Prepare data matrices
        time_matrix = np.zeros((len(algorithms), len(difficulties)))
        steps_matrix = np.zeros((len(algorithms), len(difficulties)))
        success_matrix = np.zeros((len(algorithms), len(difficulties)))
        efficiency_matrix = np.zeros((len(algorithms), len(difficulties)))
        
        for i, algo in enumerate(algorithms):
            for j, diff in enumerate(difficulties):
                key = f"{algo}_{diff}"
                if key in benchmark_data:
                    data = benchmark_data[key]
                    time_matrix[i, j] = data['execution_times']['mean']
                    steps_matrix[i, j] = data['steps_taken']['mean']
                    success_matrix[i, j] = data['success_rate']
                    efficiency_matrix[i, j] = data['efficiency']['mean_steps_per_second']
        
        # Plot 1: Average Execution Time
        im1 = ax1.imshow(time_matrix, cmap='YlOrRd', aspect='auto')
        ax1.set_title('Average Execution Time (seconds)')
        ax1.set_xticks(range(len(difficulties)))
        ax1.set_xticklabels(difficulties)
        ax1.set_yticks(range(len(algorithms)))
        ax1.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(difficulties)):
                text = ax1.text(j, i, f'{time_matrix[i, j]:.4f}',
                               ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im1, ax=ax1)
        
        # Plot 2: Average Steps Taken
        im2 = ax2.imshow(steps_matrix, cmap='Blues', aspect='auto')
        ax2.set_title('Average Steps Taken')
        ax2.set_xticks(range(len(difficulties)))
        ax2.set_xticklabels(difficulties)
        ax2.set_yticks(range(len(algorithms)))
        ax2.set_yticklabels(algorithms)
        
        for i in range(len(algorithms)):
            for j in range(len(difficulties)):
                text = ax2.text(j, i, f'{steps_matrix[i, j]:.0f}',
                               ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im2, ax=ax2)
        
        # Plot 3: Success Rate
        im3 = ax3.imshow(success_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        ax3.set_title('Success Rate')
        ax3.set_xticks(range(len(difficulties)))
        ax3.set_xticklabels(difficulties)
        ax3.set_yticks(range(len(algorithms)))
        ax3.set_yticklabels(algorithms)
        
        for i in range(len(algorithms)):
            for j in range(len(difficulties)):
                text = ax3.text(j, i, f'{success_matrix[i, j]:.1%}',
                               ha="center", va="center", color="black", fontsize=10)
        
        plt.colorbar(im3, ax=ax3)
        
        # Plot 4: Algorithm Efficiency
        im4 = ax4.imshow(efficiency_matrix, cmap='viridis', aspect='auto')
        ax4.set_title('Algorithm Efficiency (steps/second)')
        ax4.set_xticks(range(len(difficulties)))
        ax4.set_xticklabels(difficulties)
        ax4.set_yticks(range(len(algorithms)))
        ax4.set_yticklabels(algorithms)
        
        for i in range(len(algorithms)):
            for j in range(len(difficulties)):
                text = ax4.text(j, i, f'{efficiency_matrix[i, j]:.0f}',
                               ha="center", va="center", color="white", fontsize=10)
        
        plt.colorbar(im4, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def plot_time_distribution(self, timing_data: Dict[str, List[float]],
                              title: str = "Execution Time Distribution",
                              save_path: str = None) -> plt.Figure:
        """
        Plot distribution of execution times for different algorithms.
        
        Args:
            timing_data: Dictionary mapping algorithm names to lists of execution times
            title: Chart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=self.config.title_font_size, fontweight='bold')
        
        # Box plot
        data_for_box = list(timing_data.values())
        labels = list(timing_data.keys())
        colors = [self.colors.get(label.lower(), '#7F8C8D') for label in labels]
        
        box_plot = ax1.boxplot(data_for_box, labels=labels, patch_artist=True)
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_title('Box Plot of Execution Times')
        ax1.set_ylabel('Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Histogram/Distribution plot
        for i, (label, times) in enumerate(timing_data.items()):
            color = colors[i]
            ax2.hist(times, alpha=0.6, label=label, color=color, bins=10)
        
        ax2.set_title('Distribution of Execution Times')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def plot_scalability_analysis(self, scalability_data: Dict[str, Dict],
                                 title: str = "Algorithm Scalability Analysis",
                                 save_path: str = None) -> plt.Figure:
        """
        Plot scalability analysis showing how algorithms perform with increasing complexity.
        
        Args:
            scalability_data: Dictionary with scalability analysis results
            title: Chart title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle(title, fontsize=self.config.title_font_size, fontweight='bold')
        
        for algo_name, data in scalability_data.items():
            if 'scalability_results' in data:
                results = data['scalability_results']
                
                # Extract complexity indicators and performance metrics
                complexities = []
                times = []
                steps = []
                
                for complexity_level, metrics in results.items():
                    complexities.append(metrics['complexity_indicator'])
                    times.append(metrics['avg_time'])
                    steps.append(metrics['avg_steps'])
                
                # Sort by complexity
                sorted_data = sorted(zip(complexities, times, steps))
                complexities, times, steps = zip(*sorted_data)
                
                color = self.colors.get(algo_name.lower(), '#7F8C8D')
                
                # Time vs Complexity
                ax1.plot(complexities, times, 'o-', label=algo_name, color=color, linewidth=2)
                ax1.set_xlabel('Problem Complexity (empty cells)')
                ax1.set_ylabel('Average Time (seconds)')
                ax1.set_title('Time vs Problem Complexity')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Steps vs Complexity
                ax2.plot(complexities, steps, 's-', label=algo_name, color=color, linewidth=2)
                ax2.set_xlabel('Problem Complexity (empty cells)')
                ax2.set_ylabel('Average Steps')
                ax2.set_title('Steps vs Problem Complexity')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def plot_puzzle_board(self, board: Board, title: str = "Sudoku Puzzle",
                         highlight_cells: List[Tuple[int, int]] = None,
                         save_path: str = None) -> plt.Figure:
        """
        Visualize a Sudoku board with optional cell highlighting.
        
        Args:
            board: The Sudoku board to visualize
            title: Title for the plot
            highlight_cells: List of (row, col) tuples to highlight
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Create the grid
        for i in range(10):
            linewidth = 2 if i % 3 == 0 else 1
            ax.axhline(i, color='black', linewidth=linewidth)
            ax.axvline(i, color='black', linewidth=linewidth)
        
        # Fill in the numbers
        for row in range(9):
            for col in range(9):
                value = board.get_cell(row, col)
                if value != 0:
                    ax.text(col + 0.5, 8.5 - row, str(value), 
                           fontsize=16, ha='center', va='center', fontweight='bold')
        
        # Highlight specified cells
        if highlight_cells:
            for row, col in highlight_cells:
                rect = patches.Rectangle((col, 8 - row), 1, 1, 
                                       linewidth=2, edgecolor='red', 
                                       facecolor='yellow', alpha=0.3)
                ax.add_patch(rect)
        
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 9)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=self.config.title_font_size, fontweight='bold')
        ax.axis('off')
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def create_performance_dashboard(self, comparison_data: Dict[str, Any],
                                   title: str = "Sudoku Solver Performance Dashboard",
                                   save_path: str = None) -> plt.Figure:
        """
        Create a comprehensive performance dashboard with multiple visualizations.
        
        Args:
            comparison_data: Complete comparison data from PerformanceAnalyzer
            title: Dashboard title
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)
        
        # Create a grid layout
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Extract algorithms and difficulties
        algorithms = comparison_data.get('algorithms', [])
        difficulties = comparison_data.get('difficulty_levels', [])
        detailed_results = comparison_data.get('detailed_results', {})
        
        # Plot 1: Time comparison by difficulty (top-left)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_time_by_difficulty(ax1, detailed_results, algorithms, difficulties)
        
        # Plot 2: Steps comparison by difficulty (top-right)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_steps_by_difficulty(ax2, detailed_results, algorithms, difficulties)
        
        # Plot 3: Success rate heatmap (middle-left)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_success_rate_heatmap(ax3, detailed_results, algorithms, difficulties)
        
        # Plot 4: Efficiency comparison (middle-right)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_efficiency_comparison(ax4, detailed_results, algorithms, difficulties)
        
        # Plot 5: Statistical summary (bottom)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_statistical_summary(ax5, detailed_results, algorithms, difficulties)
        
        if save_path:
            plt.savefig(save_path, format=self.config.save_format, dpi=self.config.dpi,
                       bbox_inches='tight')
        
        return fig
    
    def _plot_time_by_difficulty(self, ax, detailed_results, algorithms, difficulties):
        """Helper method to plot execution time by difficulty."""
        width = 0.35
        x = np.arange(len(difficulties))
        
        for i, algo in enumerate(algorithms):
            times = []
            for diff in difficulties:
                key = f"{algo}_{diff}"
                if key in detailed_results:
                    times.append(detailed_results[key]['execution_times']['mean'])
                else:
                    times.append(0)
            
            color = self.colors.get(algo.lower(), '#7F8C8D')
            ax.bar(x + i * width, times, width, label=algo, color=color, alpha=0.7)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Average Time (seconds)')
        ax.set_title('Execution Time by Difficulty')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_steps_by_difficulty(self, ax, detailed_results, algorithms, difficulties):
        """Helper method to plot steps taken by difficulty."""
        width = 0.35
        x = np.arange(len(difficulties))
        
        for i, algo in enumerate(algorithms):
            steps = []
            for diff in difficulties:
                key = f"{algo}_{diff}"
                if key in detailed_results:
                    steps.append(detailed_results[key]['steps_taken']['mean'])
                else:
                    steps.append(0)
            
            color = self.colors.get(algo.lower(), '#7F8C8D')
            ax.bar(x + i * width, steps, width, label=algo, color=color, alpha=0.7)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Average Steps')
        ax.set_title('Steps Taken by Difficulty')
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(difficulties)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_success_rate_heatmap(self, ax, detailed_results, algorithms, difficulties):
        """Helper method to plot success rate heatmap."""
        success_matrix = np.zeros((len(algorithms), len(difficulties)))
        
        for i, algo in enumerate(algorithms):
            for j, diff in enumerate(difficulties):
                key = f"{algo}_{diff}"
                if key in detailed_results:
                    success_matrix[i, j] = detailed_results[key]['success_rate']
        
        im = ax.imshow(success_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_title('Success Rate Heatmap')
        ax.set_xticks(range(len(difficulties)))
        ax.set_xticklabels(difficulties)
        ax.set_yticks(range(len(algorithms)))
        ax.set_yticklabels(algorithms)
        
        # Add text annotations
        for i in range(len(algorithms)):
            for j in range(len(difficulties)):
                text = ax.text(j, i, f'{success_matrix[i, j]:.1%}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=ax)
    
    def _plot_efficiency_comparison(self, ax, detailed_results, algorithms, difficulties):
        """Helper method to plot efficiency comparison."""
        for algo in algorithms:
            efficiency_values = []
            diff_labels = []
            
            for diff in difficulties:
                key = f"{algo}_{diff}"
                if key in detailed_results:
                    efficiency = detailed_results[key]['efficiency']['mean_steps_per_second']
                    efficiency_values.append(efficiency)
                    diff_labels.append(diff)
            
            color = self.colors.get(algo.lower(), '#7F8C8D')
            ax.plot(diff_labels, efficiency_values, 'o-', label=algo, 
                   color=color, linewidth=2, markersize=8)
        
        ax.set_xlabel('Difficulty Level')
        ax.set_ylabel('Efficiency (steps/second)')
        ax.set_title('Algorithm Efficiency Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_statistical_summary(self, ax, detailed_results, algorithms, difficulties):
        """Helper method to plot statistical summary."""
        summary_data = []
        labels = []
        
        for algo in algorithms:
            for diff in difficulties:
                key = f"{algo}_{diff}"
                if key in detailed_results:
                    data = detailed_results[key]
                    summary_data.append([
                        data['execution_times']['mean'],
                        data['steps_taken']['mean'] / 1000,  # Scale down for better visualization
                        data['success_rate'] * 100,
                        data['efficiency']['mean_steps_per_second'] / 1000  # Scale down
                    ])
                    labels.append(f"{algo}\n({diff})")
        
        if summary_data:
            summary_array = np.array(summary_data).T
            
            x = np.arange(len(labels))
            width = 0.2
            
            metrics = ['Time (s)', 'Steps (k)', 'Success (%)', 'Efficiency (k/s)']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (metric, color) in enumerate(zip(metrics, colors)):
                ax.bar(x + i * width, summary_array[i], width, label=metric, 
                      color=color, alpha=0.7)
            
            ax.set_xlabel('Algorithm (Difficulty)')
            ax.set_ylabel('Normalized Values')
            ax.set_title('Statistical Summary Comparison')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)


# Utility functions for quick visualizations
def quick_comparison_plot(performance_data: Dict[str, PerformanceMetrics], 
                         save_path: str = None) -> plt.Figure:
    """
    Quick utility function to create a comparison plot.
    
    Args:
        performance_data: Dictionary mapping solver names to PerformanceMetrics
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    visualizer = SudokuVisualizationSuite()
    return visualizer.plot_performance_comparison(performance_data, save_path=save_path)


def plot_board_solution(original: Board, solution: Board, 
                       save_path: str = None) -> plt.Figure:
    """
    Plot original puzzle and its solution side by side.
    
    Args:
        original: Original puzzle board
        solution: Solved puzzle board
        save_path: Optional path to save the figure
        
    Returns:
        matplotlib Figure object
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    visualizer = SudokuVisualizationSuite()
    
    # Plot original puzzle
    fig1 = visualizer.plot_puzzle_board(original, title="Original Puzzle")
    ax1.imshow(fig1.canvas.buffer_rgba())
    ax1.set_title("Original Puzzle", fontsize=16, fontweight='bold')
    ax1.axis('off')
    
    # Plot solution
    fig2 = visualizer.plot_puzzle_board(solution, title="Solution")
    ax2.imshow(fig2.canvas.buffer_rgba())
    ax2.set_title("Solution", fontsize=16, fontweight='bold')
    ax2.axis('off')
    
    plt.close(fig1)
    plt.close(fig2)
    
    if save_path:
        plt.savefig(save_path, format='png', dpi=100, bbox_inches='tight')
    
    return fig
