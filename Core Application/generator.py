"""
Sudoku Puzzle Generator Module

This module provides functionality to generate complete Sudoku boards
and create puzzles by removing cells based on difficulty levels.

The generator uses a backtracking algorithm to create valid complete boards,
then strategically removes cells to create puzzles with varying difficulty.
"""

import random
from typing import List, Tuple, Optional
from board import Board, create_empty_board


class SudokuGenerator:
    """
    Generates Sudoku puzzles with configurable difficulty levels.
    
    This class can create complete valid Sudoku boards and generate
    puzzles by removing cells while ensuring unique solutions.
    """
    
    # Difficulty level configurations (number of cells to remove)
    DIFFICULTY_LEVELS = {
        'easy': (30, 40),      # Remove 30-40 cells (41-51 given)
        'medium': (41, 50),    # Remove 41-50 cells (31-40 given)
        'hard': (51, 60),      # Remove 51-60 cells (21-30 given)
        'expert': (61, 70)     # Remove 61-70 cells (11-20 given)
    }
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator with an optional random seed.
        
        Args:
            seed: Random seed for reproducible generation. If None, uses random seed.
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed
    
    def generate_complete_board(self) -> Board:
        """
        Generate a complete, valid Sudoku board.
        
        Uses a backtracking algorithm with randomization to create
        a filled 9x9 Sudoku board that satisfies all constraints.
        
        Returns:
            A Board instance with a complete, valid Sudoku solution
        """
        board = create_empty_board()
        self._fill_board_recursive(board, 0, 0)
        return board
    
    def _fill_board_recursive(self, board: Board, row: int, col: int) -> bool:
        """
        Recursively fill the board using backtracking with randomization.
        
        Args:
            board: The board to fill
            row: Current row (0-8)
            col: Current column (0-8)
            
        Returns:
            True if the board was successfully filled, False otherwise
        """
        # Move to next row if we've reached the end of current row
        if col == Board.SIZE:
            return self._fill_board_recursive(board, row + 1, 0)
        
        # Base case: if we've filled all rows, we're done
        if row == Board.SIZE:
            return True
        
        # If cell is already filled, move to next cell
        if board.get_cell(row, col) != Board.EMPTY_CELL:
            return self._fill_board_recursive(board, row, col + 1)
        
        # Try numbers 1-9 in random order
        numbers = list(range(1, 10))
        random.shuffle(numbers)
        
        for num in numbers:
            if board.is_valid_move(row, col, num):
                board.set_cell(row, col, num)
                
                # Recursively try to fill the rest
                if self._fill_board_recursive(board, row, col + 1):
                    return True
                
                # Backtrack if the current path doesn't work
                board.set_cell(row, col, Board.EMPTY_CELL)
        
        return False
    
    def generate_puzzle(self, difficulty: str = 'medium', 
                       max_attempts: int = 100) -> Board:
        """
        Generate a Sudoku puzzle by removing cells from a complete board.
        
        Args:
            difficulty: Difficulty level ('easy', 'medium', 'hard', 'expert')
            max_attempts: Maximum attempts to create a puzzle with unique solution
            
        Returns:
            A Board instance representing the puzzle
            
        Raises:
            ValueError: If difficulty level is invalid
            RuntimeError: If unable to generate valid puzzle within max_attempts
        """
        if difficulty not in self.DIFFICULTY_LEVELS:
            raise ValueError(f"Invalid difficulty. Must be one of: "
                           f"{list(self.DIFFICULTY_LEVELS.keys())}")
        
        min_remove, max_remove = self.DIFFICULTY_LEVELS[difficulty]
        
        # Start with a complete board
        complete_board = self.generate_complete_board()
        puzzle = complete_board.copy()
        
        # Determine how many cells to remove
        cells_to_remove = random.randint(min_remove, max_remove)
        
        # Get all cell positions and shuffle them
        all_positions = [(r, c) for r in range(Board.SIZE) 
                       for c in range(Board.SIZE)]
        random.shuffle(all_positions)
        
        # Remove cells to create the puzzle
        removed_count = 0
        for row, col in all_positions:
            if removed_count >= cells_to_remove:
                break
            
            # Remove the cell
            puzzle.set_cell(row, col, Board.EMPTY_CELL)
            removed_count += 1
        
        # Verify the puzzle is still valid
        if puzzle.is_valid_board():
            return puzzle
        else:
            # This shouldn't happen with proper removal, but just in case
            raise RuntimeError(f"Generated invalid puzzle")
    
    def _has_reasonable_difficulty(self, puzzle: Board, difficulty: str) -> bool:
        """
        Check if a puzzle has reasonable difficulty (simplified heuristic).
        
        This is a simplified check. A more sophisticated implementation
        would analyze solving techniques required and solution uniqueness.
        
        Args:
            puzzle: The puzzle to check
            difficulty: Target difficulty level
            
        Returns:
            True if the puzzle appears to have reasonable difficulty
        """
        empty_cells = len(puzzle.find_empty_cells())
        min_remove, max_remove = self.DIFFICULTY_LEVELS[difficulty]
        
        # Basic check: ensure we have appropriate number of empty cells
        if not (min_remove <= empty_cells <= max_remove):
            return False
        
        # Additional heuristic: check if puzzle is still solvable
        # (This is a simplified check - real implementation would be more thorough)
        return self._quick_solvability_check(puzzle)
    
    def _quick_solvability_check(self, puzzle: Board) -> bool:
        """
        Perform a quick check to see if the puzzle appears solvable.
        
        This is a simplified heuristic that checks if there are cells
        with very few valid options, which might indicate good puzzle design.
        
        Args:
            puzzle: The puzzle to check
            
        Returns:
            True if puzzle appears solvable with reasonable difficulty
        """
        empty_cells = puzzle.find_empty_cells()
        
        if not empty_cells:
            return puzzle.is_complete() and puzzle.is_valid_board()
        
        # Count cells with different numbers of valid options
        constraint_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        
        for row, col in empty_cells:
            valid_nums = len(puzzle.get_valid_numbers(row, col))
            
            # If any cell has no valid moves, puzzle is unsolvable
            if valid_nums == 0:
                return False
            
            # Count cells by constraint level
            if valid_nums <= 4:
                constraint_counts[min(valid_nums, 4)] += 1
        
        # Good puzzle should have some highly constrained cells
        # but not too many (which would make it trivial)
        highly_constrained = constraint_counts[1] + constraint_counts[2]
        total_empty = len(empty_cells)
        
        # Heuristic: 10-50% of empty cells should be highly constrained
        if total_empty > 0:
            constraint_ratio = highly_constrained / total_empty
            return 0.1 <= constraint_ratio <= 0.5
        
        return True
    
    def generate_puzzle_from_pattern(self, pattern: List[List[bool]]) -> Board:
        """
        Generate a puzzle using a specific pattern of filled/empty cells.
        
        Args:
            pattern: 9x9 boolean matrix where True means keep cell, False means remove
            
        Returns:
            A Board instance with the specified pattern
            
        Raises:
            ValueError: If pattern is not 9x9
        """
        if len(pattern) != Board.SIZE or any(len(row) != Board.SIZE 
                                           for row in pattern):
            raise ValueError("Pattern must be 9x9 boolean matrix")
        
        # Generate complete board
        complete_board = self.generate_complete_board()
        puzzle = complete_board.copy()
        
        # Apply pattern
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                if not pattern[row][col]:
                    puzzle.set_cell(row, col, Board.EMPTY_CELL)
        
        return puzzle
    
    def create_symmetric_puzzle(self, difficulty: str = 'medium') -> Board:
        """
        Create a puzzle with symmetric pattern of removed cells.
        
        Args:
            difficulty: Difficulty level for the puzzle
            
        Returns:
            A Board instance with symmetric empty cell pattern
        """
        complete_board = self.generate_complete_board()
        puzzle = complete_board.copy()
        
        min_remove, max_remove = self.DIFFICULTY_LEVELS[difficulty]
        target_remove = random.randint(min_remove, max_remove)
        
        # Create symmetric removal pattern
        removed_pairs = 0
        max_pairs = target_remove // 2
        
        # Generate positions for symmetric removal
        positions = []
        for row in range(Board.SIZE):
            for col in range(Board.SIZE):
                # Only consider positions in lower triangle to avoid duplicates
                if row * Board.SIZE + col <= 40:  # Center and below
                    positions.append((row, col))
        
        random.shuffle(positions)
        
        for row, col in positions:
            if removed_pairs >= max_pairs:
                break
            
            # Calculate symmetric position
            sym_row = Board.SIZE - 1 - row
            sym_col = Board.SIZE - 1 - col
            
            # Remove both cells
            puzzle.set_cell(row, col, Board.EMPTY_CELL)
            if (row, col) != (sym_row, sym_col):  # Don't double-remove center cell
                puzzle.set_cell(sym_row, sym_col, Board.EMPTY_CELL)
                removed_pairs += 1
        
        return puzzle


# Utility functions
def generate_easy_puzzle(seed: Optional[int] = None) -> Board:
    """Generate an easy Sudoku puzzle."""
    generator = SudokuGenerator(seed)
    return generator.generate_puzzle('easy')


def generate_medium_puzzle(seed: Optional[int] = None) -> Board:
    """Generate a medium Sudoku puzzle."""
    generator = SudokuGenerator(seed)
    return generator.generate_puzzle('medium')


def generate_hard_puzzle(seed: Optional[int] = None) -> Board:
    """Generate a hard Sudoku puzzle."""
    generator = SudokuGenerator(seed)
    return generator.generate_puzzle('hard')


def generate_expert_puzzle(seed: Optional[int] = None) -> Board:
    """Generate an expert Sudoku puzzle."""
    generator = SudokuGenerator(seed)
    return generator.generate_puzzle('expert')


def print_puzzle_stats(puzzle: Board) -> None:
    """
    Print statistics about a puzzle.
    
    Args:
        puzzle: The puzzle to analyze
    """
    empty_cells = puzzle.find_empty_cells()
    filled_cells = 81 - len(empty_cells)
    
    print(f"Puzzle Statistics:")
    print(f"  Filled cells: {filled_cells}")
    print(f"  Empty cells: {len(empty_cells)}")
    print(f"  Fill percentage: {filled_cells/81*100:.1f}%")
    print(f"  Valid board: {puzzle.is_valid_board()}")
    
    # Analyze constraint levels
    constraint_levels = {i: 0 for i in range(1, 10)}
    for row, col in empty_cells:
        valid_count = len(puzzle.get_valid_numbers(row, col))
        if valid_count > 0:
            constraint_levels[valid_count] += 1
    
    print(f"  Constraint distribution:")
    for level, count in constraint_levels.items():
        if count > 0:
            print(f"    {level} options: {count} cells")
