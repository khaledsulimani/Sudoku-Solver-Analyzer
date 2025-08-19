"""
Test script for the Board class to verify functionality.
"""

from board import Board, create_empty_board, create_board_from_string


def test_board_basic_functionality():
    """Test basic board operations."""
    print("=== Testing Basic Board Functionality ===")
    
    # Test empty board creation
    board = create_empty_board()
    print("Empty board created:")
    print(board)
    print()
    
    # Test setting and getting cells
    print("Testing cell operations:")
    print(f"Cell (0,0) before: {board.get_cell(0, 0)}")
    
    # Try setting a valid move
    success = board.set_cell(0, 0, 5)
    print(f"Setting (0,0) to 5: {'Success' if success else 'Failed'}")
    print(f"Cell (0,0) after: {board.get_cell(0, 0)}")
    
    # Try setting an invalid move (same row)
    success = board.set_cell(0, 1, 5)
    print(f"Setting (0,1) to 5 (invalid - same row): {'Success' if success else 'Failed'}")
    
    print("\nBoard after operations:")
    print(board)
    print()


def test_validation():
    """Test validation methods."""
    print("=== Testing Validation ===")
    
    board = create_empty_board()
    
    # Fill first row
    for i in range(9):
        board.set_cell(0, i, i + 1)
    
    print("Board with first row filled:")
    print(board)
    print()
    
    # Test valid moves
    print("Testing valid move checks:")
    print(f"Can place 1 at (1,0): {board.is_valid_move(1, 0, 1)}")  # Should be False (column conflict)
    print(f"Can place 1 at (1,1): {board.is_valid_move(1, 1, 1)}")  # Should be False (column conflict)
    print(f"Can place 1 at (3,3): {board.is_valid_move(3, 3, 1)}")  # Should be True
    
    # Test getting valid numbers
    valid_nums = board.get_valid_numbers(1, 0)
    print(f"Valid numbers for (1,0): {sorted(valid_nums)}")
    
    print(f"Board is complete: {board.is_complete()}")
    print(f"Board is valid: {board.is_valid_board()}")
    print()


def test_string_creation():
    """Test creating board from string."""
    print("=== Testing Board Creation from String ===")
    
    # Simple test pattern
    test_string = (
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
    
    board = create_board_from_string(test_string)
    print("Board created from string:")
    print(board)
    print()
    
    print(f"Board is valid: {board.is_valid_board()}")
    print(f"Number of empty cells: {len(board.find_empty_cells())}")


if __name__ == "__main__":
    test_board_basic_functionality()
    test_validation()
    test_string_creation()
    print("All tests completed!")
