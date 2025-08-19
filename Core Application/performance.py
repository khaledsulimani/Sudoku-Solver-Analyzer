"""
Performance Analysis Module

This module provides comprehensive performance measurement and analysis
for Sudoku solvers, including timing, complexity analysis, and statistical
comparisons between different solving algorithms.

Features:
- Precise timing measurements using time and timeit modules
- Statistical analysis across multiple puzzle runs
- Memory usage tracking
- Theoretical complexity analysis
- Performance profiling and bottleneck identification
"""

import time
import timeit
import statistics
import gc
import sys
from typing import List, Dict, Callable, Tuple, Any, Optional
from dataclasses import dataclass
from board import Board
from generator import SudokuGenerator
from solver_backtrack import BacktrackingSolver
from solver_dlx import DancingLinksSolver


@dataclass
class PerformanceMetrics:
    """Container for performance measurement results."""
    algorithm_name: str
    puzzle_difficulty: str
    execution_time: float
    steps_taken: int
    memory_usage: Optional[int] = None
    success_rate: float = 0.0
    additional_metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.additional_metrics is None:
            self.additional_metrics = {}


class PerformanceAnalyzer:
    """
    Comprehensive performance analyzer for Sudoku solving algorithms.
    
    This class provides tools to measure, analyze, and compare the performance
    of different Sudoku solving algorithms across various puzzle difficulties.
    """
    
    def __init__(self):
        """Initialize the performance analyzer."""
        self.results_history = []
        self.benchmark_results = {}
    
    def measure_single_solve(self, solver_func: Callable[[Board], Tuple[bool, Board]], 
                           puzzle: Board, algorithm_name: str, 
                           difficulty: str = "unknown") -> PerformanceMetrics:
        """
        Measure performance of a single solve operation.
        
        Args:
            solver_func: Function that takes a Board and returns (success, solved_board)
            puzzle: The puzzle to solve
            algorithm_name: Name of the algorithm being tested
            difficulty: Difficulty level of the puzzle
            
        Returns:
            PerformanceMetrics object with measurement results
        """
        # Garbage collection before measurement
        gc.collect()
        
        # Get initial memory usage
        initial_memory = self._get_memory_usage()
        
        # Measure execution time
        start_time = time.perf_counter()
        success, solved_board = solver_func(puzzle)
        end_time = time.perf_counter()
        
        execution_time = end_time - start_time
        
        # Get final memory usage
        final_memory = self._get_memory_usage()
        memory_delta = final_memory - initial_memory if initial_memory and final_memory else None
        
        # Extract additional metrics if the solver provides them
        additional_metrics = {}
        if hasattr(solver_func, '__self__'):
            solver_instance = solver_func.__self__
            if hasattr(solver_instance, 'get_statistics'):
                additional_metrics = solver_instance.get_statistics()
        
        steps_taken = additional_metrics.get('steps_taken', 0)
        
        metrics = PerformanceMetrics(
            algorithm_name=algorithm_name,
            puzzle_difficulty=difficulty,
            execution_time=execution_time,
            steps_taken=steps_taken,
            memory_usage=memory_delta,
            success_rate=1.0 if success else 0.0,
            additional_metrics=additional_metrics
        )
        
        self.results_history.append(metrics)
        return metrics
    
    def benchmark_algorithm(self, solver_class, puzzle_generator_func: Callable[[], Board],
                          algorithm_name: str, difficulty: str, 
                          num_puzzles: int = 10) -> Dict[str, Any]:
        """
        Benchmark an algorithm across multiple puzzles.
        
        Args:
            solver_class: Class of the solver to test
            puzzle_generator_func: Function that generates puzzles
            algorithm_name: Name of the algorithm
            difficulty: Difficulty level being tested
            num_puzzles: Number of puzzles to test
            
        Returns:
            Dictionary with comprehensive benchmark results
        """
        print(f"Benchmarking {algorithm_name} on {num_puzzles} {difficulty} puzzles...")
        
        execution_times = []
        steps_taken_list = []
        success_count = 0
        all_metrics = []
        
        for i in range(num_puzzles):
            # Generate a new puzzle
            puzzle = puzzle_generator_func()
            
            # Create fresh solver instance
            solver = solver_class()
            
            # Measure performance
            metrics = self.measure_single_solve(
                solver.solve, puzzle, algorithm_name, difficulty
            )
            
            execution_times.append(metrics.execution_time)
            steps_taken_list.append(metrics.steps_taken)
            if metrics.success_rate > 0:
                success_count += 1
            all_metrics.append(metrics)
            
            # Progress indicator
            if (i + 1) % max(1, num_puzzles // 10) == 0:
                print(f"  Progress: {i + 1}/{num_puzzles} puzzles completed")
        
        # Calculate statistics
        benchmark_results = {
            'algorithm_name': algorithm_name,
            'difficulty': difficulty,
            'num_puzzles': num_puzzles,
            'success_rate': success_count / num_puzzles,
            'execution_times': {
                'mean': statistics.mean(execution_times),
                'median': statistics.median(execution_times),
                'std_dev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
                'min': min(execution_times),
                'max': max(execution_times),
                'all_times': execution_times
            },
            'steps_taken': {
                'mean': statistics.mean(steps_taken_list),
                'median': statistics.median(steps_taken_list),
                'std_dev': statistics.stdev(steps_taken_list) if len(steps_taken_list) > 1 else 0,
                'min': min(steps_taken_list),
                'max': max(steps_taken_list),
                'all_steps': steps_taken_list
            },
            'efficiency': {
                'mean_steps_per_second': statistics.mean([
                    m.steps_taken / max(m.execution_time, 0.001) for m in all_metrics
                ]),
                'median_steps_per_second': statistics.median([
                    m.steps_taken / max(m.execution_time, 0.001) for m in all_metrics
                ])
            }
        }
        
        # Store results
        key = f"{algorithm_name}_{difficulty}"
        self.benchmark_results[key] = benchmark_results
        
        return benchmark_results
    
    def compare_algorithms(self, algorithm_configs: List[Dict], 
                         difficulty_levels: List[str] = None,
                         num_puzzles_per_test: int = 5) -> Dict[str, Any]:
        """
        Compare multiple algorithms across different difficulty levels.
        
        Args:
            algorithm_configs: List of dicts with 'name', 'class', and 'generator_func'
            difficulty_levels: List of difficulty levels to test
            num_puzzles_per_test: Number of puzzles to test per configuration
            
        Returns:
            Dictionary with comprehensive comparison results
        """
        if difficulty_levels is None:
            difficulty_levels = ['easy', 'medium', 'hard']
        
        comparison_results = {
            'algorithms': [config['name'] for config in algorithm_configs],
            'difficulty_levels': difficulty_levels,
            'detailed_results': {},
            'summary': {}
        }
        
        # Run benchmarks for each algorithm and difficulty
        for config in algorithm_configs:
            for difficulty in difficulty_levels:
                # Get appropriate puzzle generator
                generator_func = self._get_puzzle_generator(difficulty)
                
                results = self.benchmark_algorithm(
                    config['class'],
                    generator_func,
                    config['name'],
                    difficulty,
                    num_puzzles_per_test
                )
                
                key = f"{config['name']}_{difficulty}"
                comparison_results['detailed_results'][key] = results
        
        # Generate summary statistics
        comparison_results['summary'] = self._generate_comparison_summary(
            comparison_results['detailed_results']
        )
        
        return comparison_results
    
    def profile_algorithm_bottlenecks(self, solver_class, puzzle: Board, 
                                    algorithm_name: str) -> Dict[str, Any]:
        """
        Profile an algorithm to identify performance bottlenecks.
        
        Args:
            solver_class: Solver class to profile
            puzzle: Puzzle to solve
            algorithm_name: Name of the algorithm
            
        Returns:
            Dictionary with profiling results
        """
        import cProfile
        import pstats
        import io
        
        # Create solver instance
        solver = solver_class()
        
        # Profile the solve operation
        profiler = cProfile.Profile()
        profiler.enable()
        
        success, solution = solver.solve(puzzle)
        
        profiler.disable()
        
        # Analyze profiling results
        stats_stream = io.StringIO()
        stats = pstats.Stats(profiler, stream=stats_stream)
        stats.strip_dirs()
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        profiling_output = stats_stream.getvalue()
        
        return {
            'algorithm_name': algorithm_name,
            'success': success,
            'profiling_output': profiling_output,
            'solver_stats': solver.get_statistics() if hasattr(solver, 'get_statistics') else {}
        }
    
    def measure_scalability(self, solver_class, algorithm_name: str,
                          base_puzzle_size: int = 9) -> Dict[str, Any]:
        """
        Measure algorithm scalability (theoretical - Sudoku is fixed 9x9).
        
        This function analyzes how the algorithm might scale with puzzle complexity.
        
        Args:
            solver_class: Solver class to test
            algorithm_name: Name of the algorithm
            base_puzzle_size: Base puzzle size (always 9 for Sudoku)
            
        Returns:
            Dictionary with scalability analysis
        """
        # Generate puzzles with varying complexity (different difficulty levels)
        complexities = ['easy', 'medium', 'hard']
        results = {}
        
        for complexity in complexities:
            generator_func = self._get_puzzle_generator(complexity)
            
            # Test on a few puzzles of this complexity
            times = []
            steps = []
            
            for _ in range(3):
                puzzle = generator_func()
                solver = solver_class()
                
                start_time = time.perf_counter()
                success, _ = solver.solve(puzzle)
                end_time = time.perf_counter()
                
                if success:
                    times.append(end_time - start_time)
                    if hasattr(solver, 'get_statistics'):
                        stats = solver.get_statistics()
                        steps.append(stats.get('steps_taken', 0))
            
            if times:
                results[complexity] = {
                    'avg_time': statistics.mean(times),
                    'avg_steps': statistics.mean(steps) if steps else 0,
                    'complexity_indicator': len(puzzle.find_empty_cells())
                }
        
        return {
            'algorithm_name': algorithm_name,
            'scalability_results': results,
            'analysis': self._analyze_scalability_trend(results)
        }
    
    def _get_memory_usage(self) -> Optional[int]:
        """Get current memory usage in bytes."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return process.memory_info().rss
        except ImportError:
            return None
    
    def _get_puzzle_generator(self, difficulty: str) -> Callable[[], Board]:
        """Get appropriate puzzle generator function for difficulty level."""
        from generator import (generate_easy_puzzle, generate_medium_puzzle, 
                              generate_hard_puzzle, generate_expert_puzzle)
        
        generators = {
            'easy': lambda: generate_easy_puzzle(),
            'medium': lambda: generate_medium_puzzle(),
            'hard': lambda: generate_hard_puzzle(),
            'expert': lambda: generate_expert_puzzle()
        }
        
        return generators.get(difficulty, generators['medium'])
    
    def _generate_comparison_summary(self, detailed_results: Dict) -> Dict[str, Any]:
        """Generate summary statistics from detailed comparison results."""
        summary = {
            'fastest_algorithm_by_difficulty': {},
            'most_efficient_algorithm_by_difficulty': {},
            'overall_winner': None,
            'performance_rankings': {}
        }
        
        # Group results by difficulty
        by_difficulty = {}
        for key, results in detailed_results.items():
            difficulty = results['difficulty']
            if difficulty not in by_difficulty:
                by_difficulty[difficulty] = {}
            
            algorithm = results['algorithm_name']
            by_difficulty[difficulty][algorithm] = results
        
        # Find winners for each difficulty
        for difficulty, algorithms in by_difficulty.items():
            # Fastest by time
            fastest = min(algorithms.items(), 
                         key=lambda x: x[1]['execution_times']['mean'])
            summary['fastest_algorithm_by_difficulty'][difficulty] = fastest[0]
            
            # Most efficient by steps/time ratio
            most_efficient = max(algorithms.items(),
                               key=lambda x: x[1]['efficiency']['mean_steps_per_second'])
            summary['most_efficient_algorithm_by_difficulty'][difficulty] = most_efficient[0]
        
        return summary
    
    def _analyze_scalability_trend(self, scalability_results: Dict) -> Dict[str, str]:
        """Analyze scalability trends from results."""
        if len(scalability_results) < 2:
            return {'trend': 'insufficient_data'}
        
        # Sort by complexity
        sorted_results = sorted(
            scalability_results.items(),
            key=lambda x: x[1]['complexity_indicator']
        )
        
        times = [result[1]['avg_time'] for result in sorted_results]
        
        # Simple trend analysis
        if len(times) >= 2:
            if times[-1] > times[0] * 2:
                trend = 'poor_scaling'
            elif times[-1] > times[0] * 1.5:
                trend = 'moderate_scaling'
            else:
                trend = 'good_scaling'
        else:
            trend = 'unknown'
        
        return {
            'trend': trend,
            'time_increase_factor': times[-1] / times[0] if times[0] > 0 else float('inf')
        }
    
    def export_results(self, filename: str = None) -> str:
        """
        Export benchmark results to a formatted string or file.
        
        Args:
            filename: Optional filename to save results to
            
        Returns:
            Formatted results string
        """
        report = ["=== Sudoku Solver Performance Analysis Report ===\n"]
        
        # Summary of all tests
        report.append(f"Total tests performed: {len(self.results_history)}")
        report.append(f"Benchmark configurations: {len(self.benchmark_results)}\n")
        
        # Detailed benchmark results
        for key, results in self.benchmark_results.items():
            report.append(f"--- {results['algorithm_name']} ({results['difficulty']}) ---")
            report.append(f"Puzzles tested: {results['num_puzzles']}")
            report.append(f"Success rate: {results['success_rate']:.2%}")
            report.append(f"Average time: {results['execution_times']['mean']:.4f}s")
            report.append(f"Average steps: {results['steps_taken']['mean']:.0f}")
            report.append(f"Efficiency: {results['efficiency']['mean_steps_per_second']:.0f} steps/sec")
            report.append("")
        
        report_text = "\n".join(report)
        
        if filename:
            with open(filename, 'w') as f:
                f.write(report_text)
            print(f"Results exported to {filename}")
        
        return report_text


# Utility functions for quick performance testing
def quick_performance_test(puzzle: Board) -> Dict[str, PerformanceMetrics]:
    """
    Quick performance test comparing backtracking and DLX solvers.
    
    Args:
        puzzle: Puzzle to test both solvers on
        
    Returns:
        Dictionary with performance metrics for both solvers
    """
    analyzer = PerformanceAnalyzer()
    results = {}
    
    # Test backtracking solver
    bt_solver = BacktrackingSolver()
    bt_metrics = analyzer.measure_single_solve(
        bt_solver.solve, puzzle, "Backtracking", "test"
    )
    results['backtracking'] = bt_metrics
    
    # Test DLX solver
    dlx_solver = DancingLinksSolver()
    dlx_metrics = analyzer.measure_single_solve(
        dlx_solver.solve, puzzle, "Dancing Links", "test"
    )
    results['dlx'] = dlx_metrics
    
    return results


def time_function_call(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time a function call and return both result and execution time.
    
    Args:
        func: Function to time
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
        
    Returns:
        Tuple of (function_result, execution_time)
    """
    start_time = time.perf_counter()
    result = func(*args, **kwargs)
    end_time = time.perf_counter()
    
    return result, end_time - start_time


def compare_solver_performance(puzzle: Board, num_runs: int = 5) -> Dict[str, Any]:
    """
    Compare solver performance over multiple runs.
    
    Args:
        puzzle: Puzzle to solve
        num_runs: Number of runs for each solver
        
    Returns:
        Dictionary with comparison results
    """
    bt_times = []
    dlx_times = []
    
    for _ in range(num_runs):
        # Backtracking solver
        bt_solver = BacktrackingSolver()
        _, bt_time = time_function_call(bt_solver.solve, puzzle)
        bt_times.append(bt_time)
        
        # DLX solver
        dlx_solver = DancingLinksSolver()
        _, dlx_time = time_function_call(dlx_solver.solve, puzzle)
        dlx_times.append(dlx_time)
    
    return {
        'backtracking': {
            'times': bt_times,
            'average': statistics.mean(bt_times),
            'std_dev': statistics.stdev(bt_times) if len(bt_times) > 1 else 0
        },
        'dlx': {
            'times': dlx_times,
            'average': statistics.mean(dlx_times),
            'std_dev': statistics.stdev(dlx_times) if len(dlx_times) > 1 else 0
        },
        'dlx_speedup': statistics.mean(bt_times) / statistics.mean(dlx_times)
    }
