"""
Dancing Links (DLX) Sudoku Solver Module

This module implements Donald Knuth's Dancing Links algorithm (DLX) to solve
Sudoku puzzles as an exact cover problem. DLX is highly efficient for
constraint satisfaction problems like Sudoku.

The algorithm represents the Sudoku problem as a matrix where:
- Each row represents a possible move (cell, value)
- Each column represents a constraint that must be satisfied
- The goal is to find a subset of rows that covers each column exactly once

For Sudoku, the constraints are:
1. Each cell must contain exactly one number
2. Each row must contain each number exactly once  
3. Each column must contain each number exactly once
4. Each 3x3 box must contain each number exactly once
"""

import time
from typing import List, Tuple, Optional, Dict, Set
from board import Board


class DancingLinksNode:
    """
    A node in the Dancing Links data structure.
    
    Each node has four pointers (up, down, left, right) and
    can represent either a column header or a data node.
    """
    
    def __init__(self, column_header=None):
        """
        Initialize a Dancing Links node.
        
        Args:
            column_header: Reference to the column header for data nodes
        """
        self.up = self
        self.down = self
        self.left = self
        self.right = self
        self.column_header = column_header or self
        self.size = 0  # For column headers: number of nodes in column
        self.name = ""  # For column headers: constraint name
        
        # For tracking the Sudoku move this node represents
        self.row_id = -1
        self.sudoku_row = -1
        self.sudoku_col = -1
        self.sudoku_value = -1


class DancingLinksSolver:
    """
    Solves Sudoku puzzles using Dancing Links algorithm (Algorithm X).
    
    This solver converts the Sudoku problem into an exact cover problem
    and uses Knuth's DLX algorithm to find the solution efficiently.
    """
    
    def __init__(self):
        """Initialize the DLX solver with tracking variables."""
        self.steps_taken = 0
        self.nodes_created = 0
        self.start_time = 0
        self.solving_time = 0
        self.solution_found = False
        self.solution_stack = []
        
        # The dancing links matrix
        self.header = None
        self.columns = {}
        
        # Mapping from row IDs to Sudoku moves
        self.row_to_move = {}
    
    def solve(self, puzzle: Board) -> Tuple[bool, Board]:
        """
        Solve a Sudoku puzzle using Dancing Links algorithm.
        
        Args:
            puzzle: The puzzle to solve (will not be modified)
            
        Returns:
            Tuple of (success, solved_board)
        """
        # Reset tracking variables
        self.steps_taken = 0
        self.nodes_created = 0
        self.solution_found = False
        self.solution_stack = []
        self.start_time = time.time()
        
        # Work on a copy
        board_copy = puzzle.copy()
        
        # Check if initial board is valid
        if not board_copy.is_valid_board():
            self.solving_time = time.time() - self.start_time
            return False, puzzle.copy()
        
        # Build the dancing links matrix
        self._build_dlx_matrix(board_copy)
        
        # Solve using Algorithm X
        success = self._search(0)
        
        if success:
            # Apply the solution to the board
            self._apply_solution_to_board(board_copy)
        
        self.solving_time = time.time() - self.start_time
        self.solution_found = success
        
        return success, board_copy
    
    def _build_dlx_matrix(self, board: Board) -> None:
        """
        Build the Dancing Links matrix for the given Sudoku board.
        
        The matrix has 324 columns (constraints):
        - 81 cell constraints (each cell must have a value)
        - 81 row constraints (each row must have each digit 1-9)
        - 81 column constraints (each column must have each digit 1-9)
        - 81 box constraints (each 3x3 box must have each digit 1-9)
        
        Args:
            board: The Sudoku board to create matrix for
        """
        # Create header node
        self.header = DancingLinksNode()
        self.header.name = "header"
        self.columns = {}
        
        # Create column headers for all constraints
        self._create_column_headers()
        
        # Create rows for all possible moves
        row_id = 0
        for row in range(9):
            for col in range(9):
                if board.get_cell(row, col) == Board.EMPTY_CELL:
                    # For empty cells, create rows for all valid values
                    valid_numbers = board.get_valid_numbers(row, col)
                    for value in valid_numbers:
                        self._create_matrix_row(row_id, row, col, value)
                        row_id += 1
                else:
                    # For filled cells, create row for the existing value
                    value = board.get_cell(row, col)
                    self._create_matrix_row(row_id, row, col, value)
                    row_id += 1
    
    def _create_column_headers(self) -> None:
        """Create column headers for all Sudoku constraints."""
        constraint_names = []
        
        # Cell constraints: each cell must have exactly one value
        for row in range(9):
            for col in range(9):
                constraint_names.append(f"cell_{row}_{col}")
        
        # Row constraints: each row must have each digit exactly once
        for row in range(9):
            for digit in range(1, 10):
                constraint_names.append(f"row_{row}_{digit}")
        
        # Column constraints: each column must have each digit exactly once
        for col in range(9):
            for digit in range(1, 10):
                constraint_names.append(f"col_{col}_{digit}")
        
        # Box constraints: each 3x3 box must have each digit exactly once
        for box in range(9):
            for digit in range(1, 10):
                constraint_names.append(f"box_{box}_{digit}")
        
        # Create column header nodes
        for name in constraint_names:
            column = DancingLinksNode()
            column.name = name
            column.size = 0
            self.columns[name] = column
            self.nodes_created += 1
            
            # Link column to header
            column.left = self.header.left
            column.right = self.header
            self.header.left.right = column
            self.header.left = column
    
    def _create_matrix_row(self, row_id: int, row: int, col: int, value: int) -> None:
        """
        Create a row in the matrix representing a Sudoku move.
        
        Args:
            row_id: Unique identifier for this matrix row
            row: Sudoku row (0-8)
            col: Sudoku column (0-8)
            value: Sudoku value (1-9)
        """
        # Store the mapping from row ID to Sudoku move
        self.row_to_move[row_id] = (row, col, value)
        
        # Determine which constraints this move satisfies
        box = (row // 3) * 3 + (col // 3)
        constraint_names = [
            f"cell_{row}_{col}",
            f"row_{row}_{value}",
            f"col_{col}_{value}",
            f"box_{box}_{value}"
        ]
        
        # Create nodes for each constraint
        nodes = []
        for constraint_name in constraint_names:
            if constraint_name in self.columns:
                node = DancingLinksNode(self.columns[constraint_name])
                node.row_id = row_id
                node.sudoku_row = row
                node.sudoku_col = col
                node.sudoku_value = value
                nodes.append(node)
                self.nodes_created += 1
                
                # Link node into its column
                column = self.columns[constraint_name]
                node.up = column.up
                node.down = column
                column.up.down = node
                column.up = node
                column.size += 1
        
        # Link nodes horizontally to form a row
        if len(nodes) > 1:
            for i in range(len(nodes)):
                nodes[i].left = nodes[i-1]
                nodes[i].right = nodes[(i+1) % len(nodes)]
    
    def _search(self, k: int) -> bool:
        """
        Recursive search function implementing Algorithm X.
        
        Args:
            k: Current depth of search
            
        Returns:
            True if solution found, False otherwise
        """
        self.steps_taken += 1
        
        # If matrix is empty, we found a solution
        if self.header.right == self.header:
            return True
        
        # Choose column with minimum size (MRV heuristic)
        column = self._choose_column()
        
        # Cover the chosen column
        self._cover_column(column)
        
        # Try each row in this column
        row_node = column.down
        while row_node != column:
            # Add this row to solution
            self.solution_stack.append(row_node)
            
            # Cover all other columns in this row
            j = row_node.right
            while j != row_node:
                self._cover_column(j.column_header)
                j = j.right
            
            # Recursively search
            if self._search(k + 1):
                return True
            
            # Backtrack: uncover columns and remove row from solution
            self.solution_stack.pop()
            j = row_node.left
            while j != row_node:
                self._uncover_column(j.column_header)
                j = j.left
            
            row_node = row_node.down
        
        # Uncover the column before backtracking
        self._uncover_column(column)
        return False
    
    def _choose_column(self) -> DancingLinksNode:
        """
        Choose column with minimum size (Most Constrained Variable heuristic).
        
        Returns:
            Column header node with minimum size
        """
        min_size = float('inf')
        chosen_column = None
        
        column = self.header.right
        while column != self.header:
            if column.size < min_size:
                min_size = column.size
                chosen_column = column
            column = column.right
        
        return chosen_column
    
    def _cover_column(self, column: DancingLinksNode) -> None:
        """
        Cover a column in the dancing links matrix.
        
        Args:
            column: Column header to cover
        """
        # Remove column header from header list
        column.right.left = column.left
        column.left.right = column.right
        
        # Remove all rows in this column
        i = column.down
        while i != column:
            j = i.right
            while j != i:
                # Remove j from its column
                j.down.up = j.up
                j.up.down = j.down
                j.column_header.size -= 1
                j = j.right
            i = i.down
    
    def _uncover_column(self, column: DancingLinksNode) -> None:
        """
        Uncover a column in the dancing links matrix.
        
        Args:
            column: Column header to uncover
        """
        # Restore all rows in this column
        i = column.up
        while i != column:
            j = i.left
            while j != i:
                # Restore j to its column
                j.column_header.size += 1
                j.down.up = j
                j.up.down = j
                j = j.left
            i = i.up
        
        # Restore column header to header list
        column.right.left = column
        column.left.right = column
    
    def _apply_solution_to_board(self, board: Board) -> None:
        """
        Apply the found solution to the board.
        
        Args:
            board: Board to apply solution to
        """
        for node in self.solution_stack:
            row_id = node.row_id
            if row_id in self.row_to_move:
                row, col, value = self.row_to_move[row_id]
                board.set_cell(row, col, value)
    
    def get_statistics(self) -> dict:
        """
        Get solving statistics.
        
        Returns:
            Dictionary with solving statistics
        """
        return {
            'steps_taken': self.steps_taken,
            'nodes_created': self.nodes_created,
            'solving_time': self.solving_time,
            'solution_found': self.solution_found,
            'matrix_size': f"{self.nodes_created} nodes",
            'efficiency': self.steps_taken / max(self.solving_time, 0.001) if self.solving_time > 0 else 0
        }
    
    def reset_statistics(self) -> None:
        """Reset all solving statistics."""
        self.steps_taken = 0
        self.nodes_created = 0
        self.solving_time = 0
        self.solution_found = False
        self.solution_stack = []


class DLXSudokuAnalyzer:
    """
    Analyzes DLX solver performance and provides insights.
    """
    
    @staticmethod
    def analyze_matrix_complexity(puzzle: Board) -> dict:
        """
        Analyze the complexity of the DLX matrix for a puzzle.
        
        Args:
            puzzle: The puzzle to analyze
            
        Returns:
            Dictionary with matrix complexity metrics
        """
        empty_cells = puzzle.find_empty_cells()
        filled_cells = 81 - len(empty_cells)
        
        # Estimate matrix dimensions
        total_rows = 0
        constraint_satisfaction = 0
        
        for row in range(9):
            for col in range(9):
                if puzzle.get_cell(row, col) == Board.EMPTY_CELL:
                    valid_moves = len(puzzle.get_valid_numbers(row, col))
                    total_rows += valid_moves
                    constraint_satisfaction += valid_moves
                else:
                    total_rows += 1
                    constraint_satisfaction += 1
        
        matrix_density = constraint_satisfaction / (total_rows * 324) if total_rows > 0 else 0
        
        return {
            'empty_cells': len(empty_cells),
            'filled_cells': filled_cells,
            'estimated_matrix_rows': total_rows,
            'matrix_columns': 324,
            'estimated_density': matrix_density,
            'complexity_score': total_rows * 324 * matrix_density
        }


# Utility functions
def solve_with_dlx(puzzle: Board) -> Tuple[bool, Board]:
    """
    Quick utility function to solve a puzzle with DLX.
    
    Args:
        puzzle: The puzzle to solve
        
    Returns:
        Tuple of (success, solved_board)
    """
    solver = DancingLinksSolver()
    return solver.solve(puzzle)


def compare_dlx_with_backtrack(puzzle: Board) -> dict:
    """
    Compare DLX solver with backtracking solver.
    
    Args:
        puzzle: The puzzle to solve with both methods
        
    Returns:
        Dictionary comparing both solvers
    """
    from solver_backtrack import BacktrackingSolver
    
    # Test DLX solver
    dlx_solver = DancingLinksSolver()
    dlx_success, _ = dlx_solver.solve(puzzle)
    dlx_stats = dlx_solver.get_statistics()
    
    # Test backtracking solver
    bt_solver = BacktrackingSolver()
    bt_success, _ = bt_solver.solve(puzzle)
    bt_stats = bt_solver.get_statistics()
    
    return {
        'dlx': {
            'success': dlx_success,
            'steps': dlx_stats['steps_taken'],
            'time': dlx_stats['solving_time'],
            'nodes_created': dlx_stats['nodes_created']
        },
        'backtrack': {
            'success': bt_success,
            'steps': bt_stats['steps_taken'],
            'time': bt_stats['solving_time'],
            'backtracks': bt_stats['backtrack_count']
        },
        'comparison': {
            'dlx_faster': dlx_stats['solving_time'] < bt_stats['solving_time'],
            'time_ratio': bt_stats['solving_time'] / max(dlx_stats['solving_time'], 0.001),
            'steps_ratio': bt_stats['steps_taken'] / max(dlx_stats['steps_taken'], 1)
        }
    }
