"""
Test script for the Dancing Links (DLX) Solver.
"""

from solver_dlx import (DancingLinksSolver, DLXSudokuAnalyzer, 
                       solve_with_dlx, compare_dlx_with_backtrack)
from generator import generate_easy_puzzle, generate_medium_puzzle
from board import create_board_from_string


def test_basic_dlx_solving():
    """Test basic DLX solving functionality."""
    print("=== Testing Basic DLX Solving ===")
    
    # Create a test puzzle
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
    
    # Solve with DLX
    solver = DancingLinksSolver()
    success, solution = solver.solve(puzzle)
    
    print(f"DLX solving success: {success}")
    if success:
        print("Solution:")
        print(solution)
        print()
        print(f"Solution is valid: {solution.is_valid_board()}")
        print(f"Solution is complete: {solution.is_complete()}")
    
    # Print DLX statistics
    stats = solver.get_statistics()
    print("\nDLX Solving Statistics:")
    for key, value in stats.items():
        if key == 'solving_time':
            print(f"  {key}: {value:.4f} seconds")
        elif key == 'efficiency':
            print(f"  {key}: {value:.2f} steps/second")
        else:
            print(f"  {key}: {value}")
    print()


def test_dlx_vs_backtrack():
    """Test comparison between DLX and backtracking solvers."""
    print("=== Testing DLX vs Backtracking Comparison ===")
    
    # Test on multiple puzzles
    test_puzzles = [
        ("Easy", generate_easy_puzzle(seed=42)),
        ("Medium", generate_medium_puzzle(seed=42))
    ]
    
    for name, puzzle in test_puzzles:
        print(f"\n--- {name} Puzzle Comparison ---")
        print("Puzzle:")
        print(puzzle)
        print()
        
        comparison = compare_dlx_with_backtrack(puzzle)
        
        print("Results:")
        print(f"DLX Solver:")
        print(f"  Success: {comparison['dlx']['success']}")
        print(f"  Steps: {comparison['dlx']['steps']}")
        print(f"  Time: {comparison['dlx']['time']:.4f}s")
        print(f"  Nodes created: {comparison['dlx']['nodes_created']}")
        
        print(f"Backtracking Solver:")
        print(f"  Success: {comparison['backtrack']['success']}")
        print(f"  Steps: {comparison['backtrack']['steps']}")
        print(f"  Time: {comparison['backtrack']['time']:.4f}s")
        print(f"  Backtracks: {comparison['backtrack']['backtracks']}")
        
        print(f"Comparison:")
        print(f"  DLX faster: {comparison['comparison']['dlx_faster']}")
        print(f"  Time ratio (BT/DLX): {comparison['comparison']['time_ratio']:.2f}x")
        print(f"  Steps ratio (BT/DLX): {comparison['comparison']['steps_ratio']:.2f}x")


def test_matrix_complexity_analysis():
    """Test DLX matrix complexity analysis."""
    print("\n=== Testing Matrix Complexity Analysis ===")
    
    puzzles = {
        'Easy': generate_easy_puzzle(seed=123),
        'Medium': generate_medium_puzzle(seed=123)
    }
    
    analyzer = DLXSudokuAnalyzer()
    
    for name, puzzle in puzzles.items():
        analysis = analyzer.analyze_matrix_complexity(puzzle)
        print(f"\n{name} Puzzle Matrix Analysis:")
        print(f"  Empty cells: {analysis['empty_cells']}")
        print(f"  Filled cells: {analysis['filled_cells']}")
        print(f"  Estimated matrix rows: {analysis['estimated_matrix_rows']}")
        print(f"  Matrix columns: {analysis['matrix_columns']}")
        print(f"  Estimated density: {analysis['estimated_density']:.4f}")
        print(f"  Complexity score: {analysis['complexity_score']:.0f}")


def test_dlx_utility_function():
    """Test DLX utility function."""
    print("\n=== Testing DLX Utility Function ===")
    
    puzzle = generate_easy_puzzle(seed=42)
    
    print("Testing quick DLX solve...")
    success, solution = solve_with_dlx(puzzle)
    
    print(f"Quick DLX solve success: {success}")
    if success:
        print(f"Solution is valid: {solution.is_valid_board()}")
        print(f"Solution is complete: {solution.is_complete()}")


def test_edge_cases():
    """Test DLX solver with edge cases."""
    print("\n=== Testing Edge Cases ===")
    
    # Test with already complete puzzle
    complete_puzzle_string = (
        "534678912"
        "672195348"
        "198342567"
        "859761423"
        "426853791"
        "713924856"
        "961537284"
        "287419635"
        "345286179"
    )
    
    complete_puzzle = create_board_from_string(complete_puzzle_string)
    
    print("Testing with complete puzzle:")
    success, solution = solve_with_dlx(complete_puzzle)
    print(f"Success: {success}")
    print(f"Solution equals input: {solution == complete_puzzle}")
    
    # Test with minimal clues puzzle
    minimal_puzzle_string = (
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
    
    minimal_puzzle = create_board_from_string(minimal_puzzle_string)
    
    print("\nTesting with minimal clues puzzle:")
    solver = DancingLinksSolver()
    success, solution = solver.solve(minimal_puzzle)
    print(f"Success: {success}")
    if success:
        print(f"Valid solution: {solution.is_valid_board()}")
        print(f"Complete solution: {solution.is_complete()}")
    
    stats = solver.get_statistics()
    print(f"Steps taken: {stats['steps_taken']}")
    print(f"Nodes created: {stats['nodes_created']}")


if __name__ == "__main__":
    test_basic_dlx_solving()
    test_dlx_vs_backtrack()
    test_matrix_complexity_analysis()
    test_dlx_utility_function()
    test_edge_cases()
    print("\nAll DLX solver tests completed!")
