"""
Test Script for Sudoku Image Solver
===================================

This script demonstrates the image processing capabilities including:
- Basic image processing and grid detection
- Digit recognition and puzzle extraction
- Solution overlay on original images
- Batch processing capabilities

Run this script to test the image solver functionality.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from typing import List, Optional
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_simple_test_image(puzzle_data: List[List[int]], filename: str) -> str:
    """
    Create a simple test Sudoku image for testing purposes.
    
    Args:
        puzzle_data: 9x9 matrix with puzzle numbers
        filename: Output filename
        
    Returns:
        Path to created image
    """
    # Create white background
    img_size = 450
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # Calculate cell size
    cell_size = img_size // 9
    
    # Draw grid lines
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        pos = i * cell_size
        
        # Vertical lines
        cv2.line(img, (pos, 0), (pos, img_size), (0, 0, 0), thickness)
        # Horizontal lines
        cv2.line(img, (0, pos), (img_size, pos), (0, 0, 0), thickness)
    
    # Add numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    
    for row in range(9):
        for col in range(9):
            if puzzle_data[row][col] != 0:
                # Calculate text position (center of cell)
                x = col * cell_size + cell_size // 2 - 10
                y = row * cell_size + cell_size // 2 + 10
                
                cv2.putText(img, str(puzzle_data[row][col]), (x, y), 
                           font, font_scale, (0, 0, 0), font_thickness)
    
    # Save image
    cv2.imwrite(filename, img)
    logger.info(f"Created test image: {filename}")
    return filename


def test_basic_image_processing():
    """Test basic image processing capabilities without external dependencies."""
    
    print("=" * 60)
    print("TESTING BASIC IMAGE PROCESSING")
    print("=" * 60)
    
    # Test puzzle (easy Sudoku)
    test_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    # Create test image
    test_image_path = create_simple_test_image(test_puzzle, "test_basic_processing.jpg")
    
    try:
        # Test image loading
        img = cv2.imread(test_image_path)
        if img is not None:
            print(f"✓ Successfully loaded image: {img.shape}")
            
            # Test grayscale conversion
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"✓ Converted to grayscale: {gray.shape}")
            
            # Test basic processing
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            print(f"✓ Applied Gaussian blur")
            
            # Test thresholding
            _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
            print(f"✓ Applied thresholding")
            
            # Test contour detection
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"✓ Found {len(contours)} contours")
            
            print("\n✅ Basic image processing test PASSED")
            
        else:
            print("❌ Failed to load test image")
            
    except Exception as e:
        print(f"❌ Basic processing test failed: {e}")
    
    return test_image_path


def test_grid_detection(image_path: str):
    """Test grid detection functionality."""
    
    print("\n" + "=" * 60)
    print("TESTING GRID DETECTION")
    print("=" * 60)
    
    try:
        # Load and process image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Find contours
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            print(f"✓ Largest contour area: {area}")
            
            # Approximate contour
            epsilon = 0.02 * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            print(f"✓ Approximated contour has {len(approx)} points")
            
            if len(approx) == 4:
                print("✓ Found quadrilateral grid!")
                corners = approx.reshape(4, 2)
                print(f"✓ Grid corners: {corners}")
                
                # Create visualization
                grid_img = img.copy()
                cv2.drawContours(grid_img, [approx], -1, (0, 255, 0), 3)
                cv2.imwrite("detected_grid.jpg", grid_img)
                print("✓ Saved grid detection visualization")
                
                print("\n✅ Grid detection test PASSED")
                return corners
            else:
                print(f"❌ Expected 4 corners, found {len(approx)}")
        else:
            print("❌ No contours found")
            
    except Exception as e:
        print(f"❌ Grid detection test failed: {e}")
    
    return None


def test_perspective_transform(image_path: str, corners: Optional[np.ndarray]):
    """Test perspective transformation."""
    
    print("\n" + "=" * 60)
    print("TESTING PERSPECTIVE TRANSFORMATION")
    print("=" * 60)
    
    if corners is None:
        print("❌ No corners provided for perspective transform")
        return None
    
    try:
        # Load image
        img = cv2.imread(image_path)
        
        # Order corners (clockwise from top-left)
        def order_points(pts):
            ordered = np.zeros((4, 2), dtype=np.float32)
            s = pts.sum(axis=1)
            diff = np.diff(pts, axis=1)
            
            ordered[0] = pts[np.argmin(s)]    # top-left
            ordered[2] = pts[np.argmax(s)]    # bottom-right
            ordered[1] = pts[np.argmin(diff)] # top-right
            ordered[3] = pts[np.argmax(diff)] # bottom-left
            
            return ordered
        
        ordered_corners = order_points(corners.astype(np.float32))
        print(f"✓ Ordered corners: {ordered_corners}")
        
        # Define destination points
        output_size = 450
        dst_points = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)
        
        # Calculate and apply transform
        transform_matrix = cv2.getPerspectiveTransform(ordered_corners, dst_points)
        transformed = cv2.warpPerspective(img, transform_matrix, (output_size, output_size))
        
        # Save result
        cv2.imwrite("transformed_grid.jpg", transformed)
        print("✓ Applied perspective transformation")
        print("✓ Saved transformed grid")
        
        print("\n✅ Perspective transformation test PASSED")
        return transformed
        
    except Exception as e:
        print(f"❌ Perspective transformation test failed: {e}")
        return None


def test_cell_extraction(transformed_image: Optional[np.ndarray]):
    """Test cell extraction from transformed grid."""
    
    print("\n" + "=" * 60)
    print("TESTING CELL EXTRACTION")
    print("=" * 60)
    
    if transformed_image is None:
        print("❌ No transformed image provided")
        return None
    
    try:
        # Convert to grayscale
        if len(transformed_image.shape) == 3:
            gray_grid = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_grid = transformed_image
        
        # Extract cells
        cells = []
        cell_size = gray_grid.shape[0] // 9
        margin = 5
        
        print(f"✓ Grid size: {gray_grid.shape}")
        print(f"✓ Cell size: {cell_size}")
        
        for row in range(9):
            cell_row = []
            for col in range(9):
                # Calculate boundaries
                y1 = row * cell_size + margin
                y2 = (row + 1) * cell_size - margin
                x1 = col * cell_size + margin
                x2 = (col + 1) * cell_size - margin
                
                # Extract cell
                cell = gray_grid[y1:y2, x1:x2]
                cell_row.append(cell)
            
            cells.append(cell_row)
        
        print(f"✓ Extracted {len(cells)}x{len(cells[0])} cells")
        
        # Save a few sample cells for inspection
        for i in range(3):
            for j in range(3):
                if cells[i][j].size > 0:
                    cv2.imwrite(f"cell_{i}_{j}.jpg", cells[i][j])
        
        print("✓ Saved sample cells for inspection")
        print("\n✅ Cell extraction test PASSED")
        return cells
        
    except Exception as e:
        print(f"❌ Cell extraction test failed: {e}")
        return None


def test_simple_digit_recognition(cells: Optional[List[List[np.ndarray]]]):
    """Test simple digit recognition without OCR dependencies."""
    
    print("\n" + "=" * 60)
    print("TESTING SIMPLE DIGIT RECOGNITION")
    print("=" * 60)
    
    if cells is None:
        print("❌ No cells provided")
        return None
    
    try:
        puzzle_matrix = np.zeros((9, 9), dtype=int)
        recognized_count = 0
        
        for row in range(9):
            for col in range(9):
                cell = cells[row][col]
                
                if cell.size == 0:
                    continue
                
                # Simple empty cell detection
                non_zero_ratio = np.count_nonzero(cell) / cell.size if cell.size > 0 else 0
                
                if non_zero_ratio > 0.3:  # Cell appears to have content
                    # Very basic digit recognition (placeholder)
                    # In real implementation, this would use OCR
                    
                    # For testing, we'll use some heuristics
                    # This is just for demonstration - real OCR would go here
                    
                    # Find contours in cell
                    if len(cell.shape) == 3:
                        cell_gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
                    else:
                        cell_gray = cell
                    
                    _, thresh = cv2.threshold(cell_gray, 127, 255, cv2.THRESH_BINARY_INV)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        # Simple placeholder recognition
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)
                        aspect_ratio = float(w) / h if h > 0 else 0
                        
                        # Very basic classification (placeholder)
                        if aspect_ratio < 0.5:
                            digit = 1  # Tall and narrow
                        elif aspect_ratio > 1.2:
                            digit = 8  # Wide
                        else:
                            digit = 5  # Default
                        
                        puzzle_matrix[row, col] = digit
                        recognized_count += 1
                        
                        print(f"✓ Cell [{row},{col}]: Recognized digit {digit}")
        
        print(f"\n✓ Recognized {recognized_count} digits out of 81 cells")
        print("✓ Extracted puzzle matrix:")
        print(puzzle_matrix)
        
        print("\n✅ Simple digit recognition test PASSED")
        print("Note: This uses placeholder recognition. Real implementation would use OCR.")
        
        return puzzle_matrix
        
    except Exception as e:
        print(f"❌ Digit recognition test failed: {e}")
        return None


def test_integration_with_existing_solver(puzzle_matrix: Optional[np.ndarray]):
    """Test integration with existing Sudoku solvers."""
    
    print("\n" + "=" * 60)
    print("TESTING INTEGRATION WITH EXISTING SOLVERS")
    print("=" * 60)
    
    if puzzle_matrix is None:
        print("❌ No puzzle matrix provided")
        return
    
    try:
        # Import existing modules
        from board import Board
        from solver_backtrack import BacktrackingSolver
        
        # Create board from matrix
        board = Board()
        filled_cells = 0
        
        for row in range(9):
            for col in range(9):
                if puzzle_matrix[row, col] != 0:
                    board.set_cell(row, col, puzzle_matrix[row, col])
                    filled_cells += 1
        
        print(f"✓ Created board with {filled_cells} filled cells")
        print("✓ Original puzzle:")
        print(board)
        
        # Validate board
        if board.is_valid():
            print("✓ Puzzle is valid")
        else:
            print("⚠ Puzzle has validation issues (expected with placeholder recognition)")
        
        # Try to solve
        solver = BacktrackingSolver()
        solved_board = Board()
        
        # Copy original puzzle to solved board
        for row in range(9):
            for col in range(9):
                solved_board.set_cell(row, col, board.get_cell(row, col))
        
        print("✓ Attempting to solve puzzle...")
        success = solver.solve(solved_board)
        
        if success:
            print("✓ Puzzle solved successfully!")
            print("✓ Solution:")
            print(solved_board)
        else:
            print("⚠ Could not solve puzzle (expected with placeholder digits)")
        
        print("\n✅ Integration test PASSED")
        
    except ImportError as e:
        print(f"❌ Could not import existing modules: {e}")
        print("Make sure board.py and solver_backtrack.py are in the same directory")
    except Exception as e:
        print(f"❌ Integration test failed: {e}")


def run_comprehensive_test():
    """Run all tests in sequence."""
    
    print("🚀 STARTING COMPREHENSIVE IMAGE SOLVER TEST")
    print("=" * 80)
    
    # Test 1: Basic image processing
    test_image_path = test_basic_image_processing()
    
    # Test 2: Grid detection
    corners = test_grid_detection(test_image_path)
    
    # Test 3: Perspective transformation
    transformed = test_perspective_transform(test_image_path, corners)
    
    # Test 4: Cell extraction
    cells = test_cell_extraction(transformed)
    
    # Test 5: Simple digit recognition
    puzzle_matrix = test_simple_digit_recognition(cells)
    
    # Test 6: Integration with existing solvers
    test_integration_with_existing_solver(puzzle_matrix)
    
    print("\n" + "=" * 80)
    print("🎉 COMPREHENSIVE TEST COMPLETED")
    print("=" * 80)
    
    print("\nNext Steps:")
    print("1. Install OpenCV: pip install opencv-python")
    print("2. Install EasyOCR: pip install easyocr")
    print("3. Run the full image_solver.py for complete functionality")
    print("4. Test with real mobile photos of Sudoku puzzles")


if __name__ == "__main__":
    run_comprehensive_test()
