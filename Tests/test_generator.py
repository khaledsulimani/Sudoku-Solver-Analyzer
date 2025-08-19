"""
Test script for the Sudoku Generator to verify functionality.
"""

from generator import (SudokuGenerator, generate_easy_puzzle, 
                      generate_medium_puzzle, generate_hard_puzzle,
                      print_puzzle_stats)
from board import Board


def test_complete_board_generation():
    """Test generation of complete boards."""
    print("=== Testing Complete Board Generation ===")
    
    generator = SudokuGenerator(seed=42)  # Use seed for reproducible results
    complete_board = generator.generate_complete_board()
    
    print("Generated complete board:")
    print(complete_board)
    print()
    
    print(f"Board is complete: {complete_board.is_complete()}")
    print(f"Board is valid: {complete_board.is_valid_board()}")
    print(f"Empty cells: {len(complete_board.find_empty_cells())}")
    print()


def test_puzzle_generation():
    """Test puzzle generation with different difficulty levels."""
    print("=== Testing Puzzle Generation ===")
    
    generator = SudokuGenerator(seed=42)
    
    difficulties = ['easy', 'medium', 'hard']
    
    for difficulty in difficulties:
        print(f"\n--- {difficulty.capitalize()} Puzzle ---")
        try:
            puzzle = generator.generate_puzzle(difficulty)
            print(puzzle)
            print()
            print_puzzle_stats(puzzle)
            print()
        except Exception as e:
            print(f"Error generating {difficulty} puzzle: {e}")


def test_utility_functions():
    """Test utility functions for quick puzzle generation."""
    print("=== Testing Utility Functions ===")
    
    print("\n--- Easy Puzzle (utility function) ---")
    easy_puzzle = generate_easy_puzzle(seed=123)
    print_puzzle_stats(easy_puzzle)
    
    print("\n--- Medium Puzzle (utility function) ---")
    medium_puzzle = generate_medium_puzzle(seed=123)
    print_puzzle_stats(medium_puzzle)


def test_symmetric_puzzle():
    """Test symmetric puzzle generation."""
    print("\n=== Testing Symmetric Puzzle Generation ===")
    
    generator = SudokuGenerator(seed=42)
    symmetric_puzzle = generator.create_symmetric_puzzle('medium')
    
    print("Symmetric puzzle:")
    print(symmetric_puzzle)
    print()
    print_puzzle_stats(symmetric_puzzle)


if __name__ == "__main__":
    test_complete_board_generation()
    test_puzzle_generation()
    test_utility_functions()
    test_symmetric_puzzle()
    print("\nAll generator tests completed!")
