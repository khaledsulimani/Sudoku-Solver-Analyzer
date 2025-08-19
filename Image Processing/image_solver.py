"""
Sudoku Image Solver - Computer Vision Module
===========================================

This module extracts Sudoku puzzles from images using computer vision techniques
and integrates with the existing solver framework.

Features:
- Image preprocessing and enhancement
- Grid detection and perspective correction
- Cell segmentation and digit extraction
- OCR integration for digit recognition
- Seamless integration with existing solvers

Author: Sudoku Solver Project
Date: August 2025
"""

import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional, Union
import os
import logging
from dataclasses import dataclass

# Import existing modules
from board import Board
from solver_backtrack import BacktrackingSolver
from solver_dlx import DancingLinksSolver

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Configuration parameters for image processing pipeline."""
    
    # Image preprocessing
    gaussian_blur_ksize: int = 5
    adaptive_threshold_block_size: int = 11
    adaptive_threshold_c: int = 2
    
    # Grid detection
    min_contour_area: float = 1000
    max_contour_area: float = 500000
    approx_epsilon_factor: float = 0.02
    
    # Cell processing
    cell_margin: int = 5
    digit_threshold: float = 0.3
    
    # OCR settings
    ocr_allowlist: str = '123456789'
    ocr_confidence_threshold: float = 0.7


class SudokuImageProcessor:
    """
    Main class for processing Sudoku images and extracting puzzle data.
    
    This class handles the complete pipeline from raw image to solved puzzle:
    1. Image preprocessing and enhancement
    2. Grid detection and perspective correction
    3. Cell extraction and digit recognition
    4. Puzzle solving and solution overlay
    """
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the image processor with configuration.
        
        Args:
            config: Processing configuration parameters
        """
        self.config = config or ProcessingConfig()
        self.reader = None
        self._initialize_ocr()
        
        # Initialize solvers
        self.backtrack_solver = BacktrackingSolver()
        self.dlx_solver = DancingLinksSolver()
        
        logger.info("SudokuImageProcessor initialized successfully")
    
    def _initialize_ocr(self) -> None:
        """Initialize EasyOCR reader for digit recognition."""
        try:
            self.reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None
    
    def load_and_preprocess_image(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and apply preprocessing for grid detection.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple of (original_image, processed_image)
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Load image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        logger.info(f"Loaded image: {original.shape}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(
            gray, 
            (self.config.gaussian_blur_ksize, self.config.gaussian_blur_ksize), 
            0
        )
        
        # Apply adaptive threshold for better edge detection
        processed = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.adaptive_threshold_block_size,
            self.config.adaptive_threshold_c
        )
        
        # Invert if necessary (ensure grid lines are white on black background)
        if np.mean(processed) > 127:
            processed = cv2.bitwise_not(processed)
        
        logger.info("Image preprocessing completed")
        return original, processed
    
    def detect_sudoku_grid(self, processed_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the Sudoku grid in the processed image using contour detection.
        
        Args:
            processed_image: Preprocessed binary image
            
        Returns:
            Array of four corner points of the detected grid, or None if not found
        """
        # Find contours
        contours, _ = cv2.findContours(
            processed_image, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            logger.warning("No contours found in image")
            return None
        
        # Sort contours by area (largest first)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if (area < self.config.min_contour_area or 
                area > self.config.max_contour_area):
                continue
            
            # Approximate contour to polygon
            epsilon = self.config.approx_epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if we have a quadrilateral (4 corners)
            if len(approx) == 4:
                logger.info(f"Found grid with area: {area}")
                return self._order_points(approx.reshape(4, 2))
        
        logger.warning("No suitable grid contour found")
        return None
    
    def _order_points(self, points: np.ndarray) -> np.ndarray:
        """
        Order points in clockwise order: top-left, top-right, bottom-right, bottom-left.
        
        Args:
            points: Array of 4 points
            
        Returns:
            Ordered array of points
        """
        # Initialize ordered points array
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = points.sum(axis=1)
        diff = np.diff(points, axis=1)
        
        # Top-left: smallest sum
        # Bottom-right: largest sum
        ordered[0] = points[np.argmin(s)]  # top-left
        ordered[2] = points[np.argmax(s)]  # bottom-right
        
        # Top-right: smallest difference (x - y)
        # Bottom-left: largest difference (x - y)
        ordered[1] = points[np.argmin(diff)]  # top-right
        ordered[3] = points[np.argmax(diff)]  # bottom-left
        
        return ordered
    
    def apply_perspective_transform(self, image: np.ndarray, corners: np.ndarray, 
                                  output_size: int = 450) -> np.ndarray:
        """
        Apply perspective transformation to get a top-down view of the grid.
        
        Args:
            image: Input image
            corners: Four corner points of the grid
            output_size: Size of the output square image
            
        Returns:
            Transformed image with corrected perspective
        """
        # Define destination points (square grid)
        dst_points = np.array([
            [0, 0],
            [output_size - 1, 0],
            [output_size - 1, output_size - 1],
            [0, output_size - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        transform_matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(image, transform_matrix, (output_size, output_size))
        
        logger.info(f"Applied perspective transform: {output_size}x{output_size}")
        return transformed
    
    def extract_cells(self, grid_image: np.ndarray) -> List[List[np.ndarray]]:
        """
        Extract individual cells from the transformed grid image.
        
        Args:
            grid_image: Transformed grid image (450x450)
            
        Returns:
            9x9 list of cell images
        """
        cells = []
        cell_size = grid_image.shape[0] // 9
        
        for row in range(9):
            cell_row = []
            for col in range(9):
                # Calculate cell boundaries
                y1 = row * cell_size
                y2 = (row + 1) * cell_size
                x1 = col * cell_size
                x2 = (col + 1) * cell_size
                
                # Extract cell with margin
                margin = self.config.cell_margin
                cell = grid_image[y1 + margin:y2 - margin, x1 + margin:x2 - margin]
                
                # Ensure cell is not empty
                if cell.size > 0:
                    cell_row.append(cell)
                else:
                    # Create empty cell if extraction failed
                    cell_row.append(np.zeros((cell_size - 2*margin, cell_size - 2*margin), dtype=np.uint8))
            
            cells.append(cell_row)
        
        logger.info("Extracted 81 cells from grid")
        return cells
    
    def recognize_digit(self, cell_image: np.ndarray) -> int:
        """
        Recognize digit in a cell image using OCR.
        
        Args:
            cell_image: Image of a single cell
            
        Returns:
            Recognized digit (1-9) or 0 if empty/unrecognized
        """
        if self.reader is None:
            logger.warning("OCR reader not initialized")
            return 0
        
        # Check if cell is mostly empty
        if self._is_cell_empty(cell_image):
            return 0
        
        try:
            # Preprocess cell for better OCR
            processed_cell = self._preprocess_cell_for_ocr(cell_image)
            
            # Perform OCR
            results = self.reader.readtext(processed_cell)
            
            for (bbox, text, confidence) in results:
                # Filter results
                if (confidence > self.config.ocr_confidence_threshold and 
                    text.isdigit() and 
                    '1' <= text <= '9'):
                    return int(text)
            
            return 0
            
        except Exception as e:
            logger.debug(f"OCR failed for cell: {e}")
            return 0
    
    def _is_cell_empty(self, cell_image: np.ndarray) -> bool:
        """
        Check if a cell appears to be empty based on pixel density.
        
        Args:
            cell_image: Image of a single cell
            
        Returns:
            True if cell appears empty
        """
        if cell_image.size == 0:
            return True
        
        # Convert to binary if needed
        if len(cell_image.shape) == 3:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate the ratio of non-black pixels
        non_zero_ratio = np.count_nonzero(cell_image) / cell_image.size
        
        return non_zero_ratio < self.config.digit_threshold
    
    def _preprocess_cell_for_ocr(self, cell_image: np.ndarray) -> np.ndarray:
        """
        Preprocess individual cell for better OCR accuracy.
        
        Args:
            cell_image: Raw cell image
            
        Returns:
            Preprocessed cell image
        """
        # Convert to grayscale if needed
        if len(cell_image.shape) == 3:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        
        # Resize for better OCR (OCR works better on larger images)
        height, width = cell_image.shape
        if height < 50 or width < 50:
            scale_factor = max(50 / height, 50 / width)
            new_height = int(height * scale_factor)
            new_width = int(width * scale_factor)
            cell_image = cv2.resize(cell_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # Apply Gaussian blur to smooth edges
        cell_image = cv2.GaussianBlur(cell_image, (3, 3), 0)
        
        # Apply threshold to get clean binary image
        _, cell_image = cv2.threshold(cell_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up the image
        kernel = np.ones((2, 2), np.uint8)
        cell_image = cv2.morphologyEx(cell_image, cv2.MORPH_CLOSE, kernel)
        
        return cell_image
    
    def extract_puzzle_matrix(self, image_path: str) -> Optional[np.ndarray]:
        """
        Complete pipeline to extract Sudoku puzzle matrix from image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            9x9 numpy array representing the puzzle, or None if extraction failed
        """
        try:
            # Step 1: Load and preprocess image
            original, processed = self.load_and_preprocess_image(image_path)
            
            # Step 2: Detect grid
            corners = self.detect_sudoku_grid(processed)
            if corners is None:
                logger.error("Could not detect Sudoku grid in image")
                return None
            
            # Step 3: Apply perspective transform
            transformed = self.apply_perspective_transform(original, corners)
            
            # Step 4: Extract cells
            cells = self.extract_cells(transformed)
            
            # Step 5: Recognize digits
            puzzle_matrix = np.zeros((9, 9), dtype=int)
            
            for row in range(9):
                for col in range(9):
                    digit = self.recognize_digit(cells[row][col])
                    puzzle_matrix[row, col] = digit
            
            logger.info("Successfully extracted puzzle matrix")
            logger.info(f"Puzzle preview:\n{puzzle_matrix}")
            
            return puzzle_matrix
            
        except Exception as e:
            logger.error(f"Failed to extract puzzle matrix: {e}")
            return None
    
    def solve_from_image(self, image_path: str, algorithm: str = 'backtrack') -> Optional[Tuple[Board, Board]]:
        """
        Complete pipeline: extract puzzle from image and solve it.
        
        Args:
            image_path: Path to the input image
            algorithm: Solving algorithm ('backtrack' or 'dlx')
            
        Returns:
            Tuple of (original_board, solved_board) or None if failed
        """
        # Extract puzzle matrix
        puzzle_matrix = self.extract_puzzle_matrix(image_path)
        if puzzle_matrix is None:
            return None
        
        # Create board from matrix
        original_board = Board()
        for row in range(9):
            for col in range(9):
                if puzzle_matrix[row, col] != 0:
                    original_board.set_cell(row, col, puzzle_matrix[row, col])
        
        # Validate the extracted puzzle
        if not original_board.is_valid():
            logger.error("Extracted puzzle is invalid")
            return None
        
        # Solve the puzzle
        solver = self.dlx_solver if algorithm == 'dlx' else self.backtrack_solver
        
        solved_board = Board()
        for row in range(9):
            for col in range(9):
                solved_board.set_cell(row, col, original_board.get_cell(row, col))
        
        success = solver.solve(solved_board)
        
        if success:
            logger.info(f"Puzzle solved successfully using {algorithm}")
            return original_board, solved_board
        else:
            logger.error("Failed to solve extracted puzzle")
            return None
    
    def visualize_processing_steps(self, image_path: str, save_path: Optional[str] = None) -> None:
        """
        Visualize the image processing pipeline steps for debugging.
        
        Args:
            image_path: Path to the input image
            save_path: Optional path to save the visualization
        """
        try:
            # Load and process image
            original, processed = self.load_and_preprocess_image(image_path)
            
            # Detect grid
            corners = self.detect_sudoku_grid(processed)
            
            # Create visualization
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle('Sudoku Image Processing Pipeline', fontsize=16)
            
            # Original image
            axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            axes[0, 0].set_title('1. Original Image')
            axes[0, 0].axis('off')
            
            # Processed image
            axes[0, 1].imshow(processed, cmap='gray')
            axes[0, 1].set_title('2. Preprocessed')
            axes[0, 1].axis('off')
            
            # Grid detection
            grid_detection = original.copy()
            if corners is not None:
                cv2.drawContours(grid_detection, [corners.astype(int)], -1, (0, 255, 0), 3)
            axes[0, 2].imshow(cv2.cvtColor(grid_detection, cv2.COLOR_BGR2RGB))
            axes[0, 2].set_title('3. Grid Detection')
            axes[0, 2].axis('off')
            
            if corners is not None:
                # Transformed grid
                transformed = self.apply_perspective_transform(original, corners)
                axes[1, 0].imshow(cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title('4. Perspective Transform')
                axes[1, 0].axis('off')
                
                # Cell extraction
                cells = self.extract_cells(cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY))
                cell_grid = np.zeros((450, 450), dtype=np.uint8)
                for row in range(9):
                    for col in range(9):
                        y1, y2 = row * 50, (row + 1) * 50
                        x1, x2 = col * 50, (col + 1) * 50
                        if cells[row][col].size > 0:
                            resized_cell = cv2.resize(cells[row][col], (40, 40))
                            cell_grid[y1 + 5:y1 + 45, x1 + 5:x1 + 45] = resized_cell
                
                axes[1, 1].imshow(cell_grid, cmap='gray')
                axes[1, 1].set_title('5. Cell Extraction')
                axes[1, 1].axis('off')
                
                # Extracted matrix
                puzzle_matrix = self.extract_puzzle_matrix(image_path)
                if puzzle_matrix is not None:
                    axes[1, 2].imshow(puzzle_matrix, cmap='viridis')
                    axes[1, 2].set_title('6. Extracted Matrix')
                    
                    # Add numbers to the matrix visualization
                    for row in range(9):
                        for col in range(9):
                            if puzzle_matrix[row, col] != 0:
                                axes[1, 2].text(col, row, str(puzzle_matrix[row, col]), 
                                               ha='center', va='center', color='white', fontsize=12)
                    
                    axes[1, 2].set_xticks(range(9))
                    axes[1, 2].set_yticks(range(9))
                    axes[1, 2].grid(True, color='white', linewidth=2)
                else:
                    axes[1, 2].text(0.5, 0.5, 'Matrix extraction failed', 
                                  ha='center', va='center', transform=axes[1, 2].transAxes)
                    axes[1, 2].set_title('6. Extraction Failed')
            else:
                for i in range(3):
                    axes[1, i].text(0.5, 0.5, 'Grid detection failed', 
                                  ha='center', va='center', transform=axes[1, i].transAxes)
                    axes[1, i].set_title(f'{i+4}. Processing Failed')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Visualization saved to: {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Visualization failed: {e}")


def create_test_image() -> str:
    """
    Create a simple test Sudoku image for development and testing.
    
    Returns:
        Path to the created test image
    """
    # Create a simple Sudoku grid image for testing
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    
    # Draw grid lines
    for i in range(10):
        thickness = 3 if i % 3 == 0 else 1
        # Vertical lines
        cv2.line(img, (40 + i * 32, 40), (40 + i * 32, 328), (0, 0, 0), thickness)
        # Horizontal lines
        cv2.line(img, (40, 40 + i * 32), (328, 40 + i * 32), (0, 0, 0), thickness)
    
    # Add some sample numbers
    font = cv2.FONT_HERSHEY_SIMPLEX
    numbers = [
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
    
    for row in range(9):
        for col in range(9):
            if numbers[row][col] != 0:
                x = 52 + col * 32
                y = 62 + row * 32
                cv2.putText(img, str(numbers[row][col]), (x, y), font, 0.8, (0, 0, 0), 2)
    
    # Save test image
    test_path = "test_sudoku.jpg"
    cv2.imwrite(test_path, img)
    logger.info(f"Created test image: {test_path}")
    
    return test_path


if __name__ == "__main__":
    # Example usage and testing
    processor = SudokuImageProcessor()
    
    # Create test image
    test_image_path = create_test_image()
    
    # Process the test image
    print("Processing test image...")
    result = processor.solve_from_image(test_image_path)
    
    if result:
        original, solved = result
        print("\nOriginal puzzle:")
        print(original)
        print("\nSolved puzzle:")
        print(solved)
    else:
        print("Failed to process test image")
    
    # Visualize processing steps
    print("\nGenerating visualization...")
    processor.visualize_processing_steps(test_image_path, "processing_steps.png")
