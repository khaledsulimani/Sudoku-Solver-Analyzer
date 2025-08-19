"""
Test script for the Backtracking Solver.
"""

from solver_backtrack import (BacktrackingSolver, SudokuSolverAnalyzer, 
                             solve_puzzle, is_puzzle_solvable, get_solution_count)
from generator import generate_easy_puzzle, generate_medium_puzzle
from board import create_board_from_string


def test_basic_solving():
    """Test basic solving functionality."""
    print("=== Testing Basic Solving ===")
    
    # Create a simple test puzzle
    test_puzzle_string = (
        "530070000"
        "600195000"
        "098000060"
        "800060003"
        "400803001"
        "700020006"
        "060000280"
        "000419005"
        "000080079"
    )
    
    puzzle = create_board_from_string(test_puzzle_string)
    print("Original puzzle:")
    print(puzzle)
    print()
    
    # Solve the puzzle
    solver = BacktrackingSolver()
    success, solution = solver.solve(puzzle)
    
    print(f"Solving success: {success}")
    if success:
        print("Solution:")
        print(solution)
        print()
        print(f"Solution is valid: {solution.is_valid_board()}")
        print(f"Solution is complete: {solution.is_complete()}")
    
    # Print statistics
    stats = solver.get_statistics()
    print("\nSolving Statistics:")
    for key, value in stats.items():
        if key == 'solving_time':
            print(f"  {key}: {value:.4f} seconds")
        elif key == 'efficiency':
            print(f"  {key}: {value:.2f} steps/second")
        else:
            print(f"  {key}: {value}")
    print()


def test_strategy_comparison():
    """Test comparison between basic and heuristic strategies."""
    print("=== Testing Strategy Comparison ===")
    
    # Generate a medium puzzle for testing
    puzzle = generate_medium_puzzle(seed=42)
    print("Test puzzle:")
    print(puzzle)
    print()
    
    # Compare strategies
    analyzer = SudokuSolverAnalyzer()
    comparison = analyzer.compare_solving_strategies(puzzle)
    
    print("Strategy Comparison:")
    print(f"Basic Strategy:")
    print(f"  Steps: {comparison['basic_strategy']['steps']}")
    print(f"  Backtracks: {comparison['basic_strategy']['backtracks']}")
    print(f"  Time: {comparison['basic_strategy']['time']:.4f}s")
    
    print(f"Heuristic Strategy:")
    print(f"  Steps: {comparison['heuristic_strategy']['steps']}")
    print(f"  Backtracks: {comparison['heuristic_strategy']['backtracks']}")
    print(f"  Time: {comparison['heuristic_strategy']['time']:.4f}s")
    
    print(f"Improvements:")
    print(f"  Steps reduction: {comparison['improvement']['steps_reduction']:.1f}%")
    print(f"  Time reduction: {comparison['improvement']['time_reduction']:.1f}%")
    print()


def test_difficulty_analysis():
    """Test puzzle difficulty analysis."""
    print("=== Testing Difficulty Analysis ===")
    
    # Test different difficulty puzzles
    puzzles = {
        'Easy': generate_easy_puzzle(seed=123),
        'Medium': generate_medium_puzzle(seed=123)
    }
    
    analyzer = SudokuSolverAnalyzer()
    
    for name, puzzle in puzzles.items():
        analysis = analyzer.analyze_puzzle_difficulty(puzzle)
        print(f"{name} Puzzle Analysis:")
        print(f"  Detected difficulty: {analysis['difficulty']}")
        print(f"  Empty cells: {analysis['empty_cells']}")
        print(f"  Average constraints: {analysis['avg_constraints']:.2f}")
        print(f"  Highly constrained cells: {analysis['highly_constrained']}")
        print()


def test_utility_functions():
    """Test utility functions."""
    print("=== Testing Utility Functions ===")
    
    # Create a simple puzzle
    puzzle = generate_easy_puzzle(seed=42)
    
    # Test solvability check
    solvable = is_puzzle_solvable(puzzle)
    print(f"Puzzle is solvable: {solvable}")
    
    # Test solution counting
    solution_count = get_solution_count(puzzle, max_solutions=2)
    print(f"Number of solutions found: {solution_count}")
    
    # Test quick solve function
    success, solution = solve_puzzle(puzzle)
    print(f"Quick solve success: {success}")
    
    if success:
        print(f"Solution is valid: {solution.is_valid_board()}")
        print(f"Solution is complete: {solution.is_complete()}")


def test_step_by_step_solving():
    """Test step-by-step solving with callback."""
    print("\n=== Testing Step-by-Step Solving ===")
    
    # Create a very simple puzzle for demonstration
    simple_puzzle_string = (
        "100000000"
        "020000000"
        "003000000"
        "000400000"
        "000050000"
        "000006000"
        "000000700"
        "000000080"
        "000000009"
    )
    
    puzzle = create_board_from_string(simple_puzzle_string)
    
    steps = []
    
    def step_callback(board, row, col, value):
        steps.append((row, col, value))
    
    solver = BacktrackingSolver()
    success, solution = solver.solve_step_by_step(puzzle, step_callback)
    
    print(f"Step-by-step solving success: {success}")
    print(f"Total steps recorded: {len(steps)}")
    
    if steps:
        print("First few steps:")
        for i, (row, col, value) in enumerate(steps[:5]):
            print(f"  Step {i+1}: Set cell ({row},{col}) to {value}")


if __name__ == "__main__":
    test_basic_solving()
    test_strategy_comparison()
    test_difficulty_analysis()
    test_utility_functions()
    test_step_by_step_solving()
    print("All backtracking solver tests completed!")
