"""
Backtracking Sudoku Solver Module

This module implements a recursive backtracking algorithm to solve Sudoku puzzles.
The backtracking approach systematically tries all possible values for each empty cell,
backtracking when a dead end is reached.

The solver includes optimizations like:
- Most Constrained Variable (MCV) heuristic: choose cells with fewest valid options first
- Early termination on invalid boards
- Step-by-step solving for visualization
"""

from typing import List, Tuple, Optional, Set, Callable
import time
from board import Board


class BacktrackingSolver:
    """
    Solves Sudoku puzzles using recursive backtracking algorithm.
    
    This class provides multiple solving strategies and can track
    the solving process for analysis and visualization.
    """
    
    def __init__(self):
        """Initialize the solver with tracking variables."""
        self.steps_taken = 0
        self.backtrack_count = 0
        self.start_time = 0
        self.solving_time = 0
        self.solution_found = False
        
        # For step-by-step solving
        self.step_callback: Optional[Callable[[Board, int, int, int], None]] = None
    
    def solve(self, puzzle: Board, use_heuristics: bool = True) -> Tuple[bool, Board]:
        """
        Solve a Sudoku puzzle using backtracking algorithm.
        
        Args:
            puzzle: The puzzle to solve (will not be modified)
            use_heuristics: Whether to use optimization heuristics
            
        Returns:
            Tuple of (success, solved_board)
            - success: True if puzzle was solved, False if no solution exists
            - solved_board: Board with solution (copy of original if unsolved)
        """
        # Reset tracking variables
        self.steps_taken = 0
        self.backtrack_count = 0
        self.solution_found = False
        self.start_time = time.time()
        
        # Work on a copy to avoid modifying the original
        board_copy = puzzle.copy()
        
        # Check if the initial board is valid
        if not board_copy.is_valid_board():
            self.solving_time = time.time() - self.start_time
            return False, puzzle.copy()
        
        # Choose solving strategy
        if use_heuristics:
            success = self._solve_with_heuristics(board_copy)
        else:
            success = self._solve_basic(board_copy, 0, 0)
        
        self.solving_time = time.time() - self.start_time
        self.solution_found = success
        
        return success, board_copy
    
    def _solve_basic(self, board: Board, row: int, col: int) -> bool:
        """
        Basic backtracking solver without heuristics.
        
        Args:
            board: The board to solve (will be modified)
            row: Current row (0-8)
            col: Current column (0-8)
            
        Returns:
            True if solution found, False otherwise
        """
        # Move to next row if we've reached the end of current row
        if col == Board.SIZE:
            return self._solve_basic(board, row + 1, 0)
        
        # Base case: if we've processed all rows, we're done
        if row == Board.SIZE:
            return True
        
        # If cell is already filled, move to next cell
        if board.get_cell(row, col) != Board.EMPTY_CELL:
            return self._solve_basic(board, row, col + 1)
        
        # Try numbers 1-9
        for num in range(1, 10):
            self.steps_taken += 1
            
            if board.is_valid_move(row, col, num):
                board.set_cell(row, col, num)
                
                # Call step callback if provided
                if self.step_callback:
                    self.step_callback(board, row, col, num)
                
                # Recursively try to solve the rest
                if self._solve_basic(board, row, col + 1):
                    return True
                
                # Backtrack if current path doesn't work
                board.set_cell(row, col, Board.EMPTY_CELL)
                self.backtrack_count += 1
        
        return False
    
    def _solve_with_heuristics(self, board: Board) -> bool:
        """
        Solve using heuristics for better performance.
        
        Uses Most Constrained Variable (MCV) heuristic to choose
        the empty cell with fewest valid options first.
        
        Args:
            board: The board to solve (will be modified)
            
        Returns:
            True if solution found, False otherwise
        """
        # Find the most constrained empty cell
        best_cell = self._find_most_constrained_cell(board)
        
        if best_cell is None:
            # No empty cells left - puzzle is solved
            return True
        
        row, col, valid_numbers = best_cell
        
        # If no valid numbers for this cell, puzzle is unsolvable
        if not valid_numbers:
            return False
        
        # Try each valid number
        for num in valid_numbers:
            self.steps_taken += 1
            
            board.set_cell(row, col, num)
            
            # Call step callback if provided
            if self.step_callback:
                self.step_callback(board, row, col, num)
            
            # Recursively solve the rest
            if self._solve_with_heuristics(board):
                return True
            
            # Backtrack
            board.set_cell(row, col, Board.EMPTY_CELL)
            self.backtrack_count += 1
        
        return False
    
    def _find_most_constrained_cell(self, board: Board) -> Optional[Tuple[int, int, Set[int]]]:
        """
        Find the empty cell with the fewest valid options (MCV heuristic).
        
        Args:
            board: The board to analyze
            
        Returns:
            Tuple of (row, col, valid_numbers) for the most constrained cell,
            or None if no empty cells remain
        """
        best_cell = None
        min_options = 10  # More than maximum possible (9)
        
        empty_cells = board.find_empty_cells()
        
        for row, col in empty_cells:
            valid_numbers = board.get_valid_numbers(row, col)
            options_count = len(valid_numbers)
            
            # If we find a cell with only one option, use it immediately
            if options_count == 1:
                return row, col, valid_numbers
            
            # Track the cell with minimum options
            if options_count < min_options:
                min_options = options_count
                best_cell = (row, col, valid_numbers)
        
        return best_cell
    
    def solve_step_by_step(self, puzzle: Board, 
                          step_callback: Callable[[Board, int, int, int], None],
                          use_heuristics: bool = True) -> Tuple[bool, Board]:
        """
        Solve puzzle step by step, calling callback for each move.
        
        Args:
            puzzle: The puzzle to solve
            step_callback: Function called for each step (board, row, col, value)
            use_heuristics: Whether to use optimization heuristics
            
        Returns:
            Tuple of (success, solved_board)
        """
        self.step_callback = step_callback
        result = self.solve(puzzle, use_heuristics)
        self.step_callback = None
        return result
    
    def get_statistics(self) -> dict:
        """
        Get solving statistics.
        
        Returns:
            Dictionary with solving statistics
        """
        return {
            'steps_taken': self.steps_taken,
            'backtrack_count': self.backtrack_count,
            'solving_time': self.solving_time,
            'solution_found': self.solution_found,
            'efficiency': self.steps_taken / max(self.solving_time, 0.001) if self.solving_time > 0 else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset all solving statistics."""
        self.steps_taken = 0
        self.backtrack_count = 0
        self.solving_time = 0
        self.solution_found = False


class SudokuSolverAnalyzer:
    """
    Analyzes Sudoku solving performance and provides insights.
    """
    
    @staticmethod
    def analyze_puzzle_difficulty(puzzle: Board) -> dict:
        """
        Analyze puzzle difficulty by examining constraints.
        
        Args:
            puzzle: The puzzle to analyze
            
        Returns:
            Dictionary with difficulty analysis
        """
        empty_cells = puzzle.find_empty_cells()
        
        if not empty_cells:
            return {'difficulty': 'complete', 'empty_cells': 0}
        
        # Analyze constraint distribution
        constraint_counts = {i: 0 for i in range(1, 10)}
        total_constraints = 0
        
        for row, col in empty_cells:
            valid_count = len(puzzle.get_valid_numbers(row, col))
            if valid_count > 0:
                constraint_counts[valid_count] += 1
                total_constraints += valid_count
        
        # Calculate difficulty metrics
        avg_constraints = total_constraints / len(empty_cells) if empty_cells else 0
        highly_constrained = constraint_counts[1] + constraint_counts[2]
        moderately_constrained = constraint_counts[3] + constraint_counts[4]
        
        # Estimate difficulty based on constraints
        if avg_constraints >= 6:
            difficulty = 'easy'
        elif avg_constraints >= 4:
            difficulty = 'medium'
        elif avg_constraints >= 2.5:
            difficulty = 'hard'
        else:
            difficulty = 'expert'
        
        return {
            'difficulty': difficulty,
            'empty_cells': len(empty_cells),
            'avg_constraints': avg_constraints,
            'highly_constrained': highly_constrained,
            'moderately_constrained': moderately_constrained,
            'constraint_distribution': constraint_counts
        }
    
    @staticmethod
    def compare_solving_strategies(puzzle: Board) -> dict:
        """
        Compare basic vs heuristic solving strategies.
        
        Args:
            puzzle: The puzzle to solve
            
        Returns:
            Dictionary comparing both strategies
        """
        solver = BacktrackingSolver()
        
        # Solve with basic strategy
        success_basic, _ = solver.solve(puzzle, use_heuristics=False)
        stats_basic = solver.get_statistics()
        
        # Solve with heuristics
        solver.reset_statistics()
        success_heuristic, _ = solver.solve(puzzle, use_heuristics=True)
        stats_heuristic = solver.get_statistics()
        
        return {
            'basic_strategy': {
                'success': success_basic,
                'steps': stats_basic['steps_taken'],
                'backtracks': stats_basic['backtrack_count'],
                'time': stats_basic['solving_time']
            },
            'heuristic_strategy': {
                'success': success_heuristic,
                'steps': stats_heuristic['steps_taken'],
                'backtracks': stats_heuristic['backtrack_count'],
                'time': stats_heuristic['solving_time']
            },
            'improvement': {
                'steps_reduction': ((stats_basic['steps_taken'] - stats_heuristic['steps_taken']) 
                                  / max(stats_basic['steps_taken'], 1)) * 100,
                'time_reduction': ((stats_basic['solving_time'] - stats_heuristic['solving_time']) 
                                 / max(stats_basic['solving_time'], 0.001)) * 100
            }
        }


# Utility functions
def solve_puzzle(puzzle: Board, use_heuristics: bool = True) -> Tuple[bool, Board]:
    """
    Quick utility function to solve a puzzle.
    
    Args:
        puzzle: The puzzle to solve
        use_heuristics: Whether to use optimization heuristics
        
    Returns:
        Tuple of (success, solved_board)
    """
    solver = BacktrackingSolver()
    return solver.solve(puzzle, use_heuristics)


def is_puzzle_solvable(puzzle: Board) -> bool:
    """
    Check if a puzzle is solvable.
    
    Args:
        puzzle: The puzzle to check
        
    Returns:
        True if the puzzle can be solved, False otherwise
    """
    success, _ = solve_puzzle(puzzle)
    return success


def get_solution_count(puzzle: Board, max_solutions: int = 2) -> int:
    """
    Count the number of solutions for a puzzle (up to max_solutions).
    
    This is useful for checking if a puzzle has a unique solution.
    
    Args:
        puzzle: The puzzle to analyze
        max_solutions: Maximum solutions to find before stopping
        
    Returns:
        Number of solutions found (capped at max_solutions)
    """
    class SolutionCounter:
        def __init__(self, max_count):
            self.count = 0
            self.max_count = max_count
    
    counter = SolutionCounter(max_solutions)
    board_copy = puzzle.copy()
    
    def count_solutions(board: Board, row: int, col: int) -> None:
        if counter.count >= counter.max_count:
            return
        
        # Move to next row if we've reached the end of current row
        if col == Board.SIZE:
            count_solutions(board, row + 1, 0)
            return
        
        # Base case: if we've processed all rows, found a solution
        if row == Board.SIZE:
            counter.count += 1
            return
        
        # If cell is already filled, move to next cell
        if board.get_cell(row, col) != Board.EMPTY_CELL:
            count_solutions(board, row, col + 1)
            return
        
        # Try numbers 1-9
        for num in range(1, 10):
            if counter.count >= counter.max_count:
                return
            
            if board.is_valid_move(row, col, num):
                board.set_cell(row, col, num)
                count_solutions(board, row, col + 1)
                board.set_cell(row, col, Board.EMPTY_CELL)
    
    count_solutions(board_copy, 0, 0)
    return counter.count
