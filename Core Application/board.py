"""
Sudoku Board Module

This module provides the Board class that manages the Sudoku grid,
handles cell access, and performs validity checks for moves.

The Board class follows object-oriented design principles and provides
a clean interface for interacting with the Sudoku puzzle state.
"""

from typing import List, Tuple, Optional, Set
import copy


class Board:
    """
    Represents a 9x9 Sudoku board with methods for validation and manipulation.
    
    The board uses 0 to represent empty cells and integers 1-9 for filled cells.
    This class provides comprehensive validation and utility methods for
    Sudoku puzzle operations.
    """
    
    SIZE = 9  # Standard Sudoku board size
    BOX_SIZE = 3  # 3x3 sub-grid size
    EMPTY_CELL = 0  # Represents an empty cell
    
    def __init__(self, grid: Optional[List[List[int]]] = None):
        """
        Initialize a Sudoku board.
        
        Args:
            grid: Optional 9x9 grid. If None, creates an empty board.
                 Each cell should contain 0 (empty) or 1-9 (filled).
        
        Raises:
            ValueError: If the provided grid is not 9x9 or contains invalid values.
        """
        if grid is None:
            self.grid = [[self.EMPTY_CELL for _ in range(self.SIZE)] 
                        for _ in range(self.SIZE)]
        else:
            self._validate_grid(grid)
            self.grid = copy.deepcopy(grid)
    
    def _validate_grid(self, grid: List[List[int]]) -> None:
        """
        Validate that the grid has correct dimensions and values.
        
        Args:
            grid: The grid to validate.
            
        Raises:
            ValueError: If grid dimensions or values are invalid.
        """
        if len(grid) != self.SIZE:
            raise ValueError(f"Grid must have {self.SIZE} rows, got {len(grid)}")
        
        for i, row in enumerate(grid):
            if len(row) != self.SIZE:
                raise ValueError(f"Row {i} must have {self.SIZE} columns, got {len(row)}")
            
            for j, cell in enumerate(row):
                if not isinstance(cell, int) or cell < 0 or cell > 9:
                    raise ValueError(f"Invalid cell value at ({i}, {j}): {cell}. "
                                   f"Must be integer 0-9")
    
    def get_cell(self, row: int, col: int) -> int:
        """
        Get the value of a specific cell.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            
        Returns:
            The value in the cell (0 for empty, 1-9 for filled)
            
        Raises:
            IndexError: If row or col is out of bounds
        """
        self._validate_coordinates(row, col)
        return self.grid[row][col]
    
    def set_cell(self, row: int, col: int, value: int) -> bool:
        """
        Set the value of a specific cell if the move is valid.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            value: Value to set (0 for empty, 1-9 for filled)
            
        Returns:
            True if the move was valid and applied, False otherwise
            
        Raises:
            IndexError: If row or col is out of bounds
            ValueError: If value is not in range 0-9
        """
        self._validate_coordinates(row, col)
        
        if not isinstance(value, int) or value < 0 or value > 9:
            raise ValueError(f"Value must be integer 0-9, got {value}")
        
        # Check if move is valid (unless setting to empty)
        if value != self.EMPTY_CELL and not self.is_valid_move(row, col, value):
            return False
        
        self.grid[row][col] = value
        return True
    
    def is_valid_move(self, row: int, col: int, value: int) -> bool:
        """
        Check if placing a value at the given position is valid according to Sudoku rules.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            value: Value to check (1-9)
            
        Returns:
            True if the move is valid, False otherwise
            
        Raises:
            IndexError: If row or col is out of bounds
            ValueError: If value is not in range 1-9
        """
        self._validate_coordinates(row, col)
        
        if not isinstance(value, int) or value < 1 or value > 9:
            raise ValueError(f"Value must be integer 1-9, got {value}")
        
        # Check row constraint
        if value in self.grid[row]:
            return False
        
        # Check column constraint
        if value in [self.grid[r][col] for r in range(self.SIZE)]:
            return False
        
        # Check 3x3 box constraint
        box_row = (row // self.BOX_SIZE) * self.BOX_SIZE
        box_col = (col // self.BOX_SIZE) * self.BOX_SIZE
        
        for r in range(box_row, box_row + self.BOX_SIZE):
            for c in range(box_col, box_col + self.BOX_SIZE):
                if self.grid[r][c] == value:
                    return False
        
        return True
    
    def get_valid_numbers(self, row: int, col: int) -> Set[int]:
        """
        Get all valid numbers that can be placed in a specific cell.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            
        Returns:
            Set of valid numbers (1-9) that can be placed in the cell
            
        Raises:
            IndexError: If row or col is out of bounds
        """
        self._validate_coordinates(row, col)
        
        if self.grid[row][col] != self.EMPTY_CELL:
            return set()  # Cell is already filled
        
        valid_numbers = set()
        for num in range(1, 10):
            if self.is_valid_move(row, col, num):
                valid_numbers.add(num)
        
        return valid_numbers
    
    def is_complete(self) -> bool:
        """
        Check if the board is completely filled (no empty cells).
        
        Returns:
            True if no empty cells remain, False otherwise
        """
        for row in self.grid:
            if self.EMPTY_CELL in row:
                return False
        return True
    
    def is_valid_board(self) -> bool:
        """
        Check if the current board state is valid according to Sudoku rules.
        
        This method checks all filled cells to ensure no Sudoku rules are violated.
        
        Returns:
            True if the board state is valid, False otherwise
        """
        # Check each filled cell
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self.grid[row][col] != self.EMPTY_CELL:
                    # Temporarily remove the cell value and check if it's a valid move
                    original_value = self.grid[row][col]
                    self.grid[row][col] = self.EMPTY_CELL
                    
                    is_valid = self.is_valid_move(row, col, original_value)
                    
                    # Restore the original value
                    self.grid[row][col] = original_value
                    
                    if not is_valid:
                        return False
        
        return True
    
    def find_empty_cells(self) -> List[Tuple[int, int]]:
        """
        Find all empty cells on the board.
        
        Returns:
            List of (row, col) tuples representing empty cell coordinates
        """
        empty_cells = []
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                if self.grid[row][col] == self.EMPTY_CELL:
                    empty_cells.append((row, col))
        return empty_cells
    
    def copy(self) -> 'Board':
        """
        Create a deep copy of the board.
        
        Returns:
            A new Board instance with the same state
        """
        return Board(self.grid)
    
    def clear(self) -> None:
        """Clear the board (set all cells to empty)."""
        for row in range(self.SIZE):
            for col in range(self.SIZE):
                self.grid[row][col] = self.EMPTY_CELL
    
    def _validate_coordinates(self, row: int, col: int) -> None:
        """
        Validate that coordinates are within board bounds.
        
        Args:
            row: Row index
            col: Column index
            
        Raises:
            IndexError: If coordinates are out of bounds
        """
        if not (0 <= row < self.SIZE):
            raise IndexError(f"Row must be 0-{self.SIZE-1}, got {row}")
        if not (0 <= col < self.SIZE):
            raise IndexError(f"Column must be 0-{self.SIZE-1}, got {col}")
    
    def __str__(self) -> str:
        """
        Return a string representation of the board for debugging.
        
        Returns:
            A formatted string showing the board state
        """
        lines = []
        for i, row in enumerate(self.grid):
            if i % 3 == 0 and i != 0:
                lines.append("------+-------+------")
            
            row_str = ""
            for j, cell in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                
                row_str += str(cell if cell != self.EMPTY_CELL else ".") + " "
            
            lines.append(row_str.rstrip())
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Return a detailed string representation of the board."""
        return f"Board({self.grid})"
    
    def __eq__(self, other) -> bool:
        """Check equality with another Board instance."""
        if not isinstance(other, Board):
            return False
        return self.grid == other.grid


# Utility functions for working with boards
def create_empty_board() -> Board:
    """
    Create a new empty Sudoku board.
    
    Returns:
        A new Board instance with all cells empty
    """
    return Board()


def create_board_from_string(board_string: str) -> Board:
    """
    Create a board from a string representation.
    
    Args:
        board_string: 81-character string where each character is 0-9
                     (0 or '.' for empty, 1-9 for filled)
    
    Returns:
        A new Board instance
        
    Raises:
        ValueError: If string is not 81 characters or contains invalid characters
    """
    # Clean the string (remove whitespace and convert dots to zeros)
    clean_string = board_string.replace(' ', '').replace('\n', '').replace('.', '0')
    
    if len(clean_string) != 81:
        raise ValueError(f"Board string must be 81 characters, got {len(clean_string)}")
    
    grid = []
    for i in range(9):
        row = []
        for j in range(9):
            char = clean_string[i * 9 + j]
            if char.isdigit():
                row.append(int(char))
            else:
                raise ValueError(f"Invalid character '{char}' at position {i*9 + j}")
        grid.append(row)
    
    return Board(grid)
