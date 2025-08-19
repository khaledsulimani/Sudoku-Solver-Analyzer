"""
GUI Integration for Image Solver
===============================

This module integrates the image processing capabilities with the existing Pygame GUI.
Adds image loading and processing buttons to the main interface.

Features:
- Image file selection dialog
- Real-time processing visualization
- Solution overlay display
- Error handling and user feedback

Author: Sudoku Solver Project
Date: August 2025
"""

import pygame
import numpy as np
import os
from typing import Optional, Tuple, List
import threading
import queue
import time

# Import existing modules
from board import Board
from solver_backtrack import BacktrackingSolver
from solver_dlx import DancingLinksSolver

# Colors for GUI
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BLUE = (0, 100, 200)
LIGHT_BLUE = (100, 150, 255)
GREEN = (0, 200, 0)
LIGHT_GREEN = (100, 255, 100)
RED = (200, 0, 0)
ORANGE = (255, 165, 0)

class ImageSolverGUI:
    """
    GUI integration for the image solver functionality.
    Provides interface for loading images and processing Sudoku puzzles.
    """
    
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        """
        Initialize the image solver GUI integration.
        
        Args:
            screen: Pygame screen surface
            font: Font for text rendering
        """
        self.screen = screen
        self.font = font
        self.small_font = pygame.font.Font(None, 24)
        
        # Processing state
        self.current_image_path = None
        self.processing_thread = None
        self.processing_queue = queue.Queue()
        self.processing_status = "Ready"
        self.processing_progress = 0.0
        
        # Results
        self.extracted_board = None
        self.solved_board = None
        self.already_solved_board = None  # Store pre-solved board from image processor
        self.processing_error = None
        
        # Initialize solvers
        self.backtrack_solver = BacktrackingSolver()
        self.dlx_solver = DancingLinksSolver()
        
        # UI state
        self.show_image_interface = False
        self.selected_algorithm = "backtrack"
    
    def toggle_image_interface(self):
        """Toggle the image processing interface visibility."""
        self.show_image_interface = not self.show_image_interface
        if not self.show_image_interface:
            self.reset_processing_state()
    
    def reset_processing_state(self):
        """Reset all processing state variables."""
        self.current_image_path = None
        self.extracted_board = None
        self.solved_board = None
        self.already_solved_board = None
        self.processing_error = None
        self.processing_status = "Ready"
        self.processing_progress = 0.0
    
    def select_image_file(self) -> Optional[str]:
        """
        Open file browser to select an image file.
        
        Returns:
            Selected file path or None if cancelled
        """
        try:
            # Try the image browser first
            try:
                from image_browser import show_image_browser
                selected_file = show_image_browser(self.screen, self.font)
                
                if selected_file and selected_file != "CANCEL":
                    print(f"User selected file via browser: {selected_file}")
                    return selected_file
                elif selected_file == "CANCEL":
                    print("User cancelled file selection")
                    return None
            except ImportError:
                print("Image browser not available, using fallback methods")
            
            import os
            import subprocess
            import platform
            
            # For testing, provide some default sample paths
            sample_files = [
                "test_sudoku.jpg",
                "demo_sudoku_1.jpg", 
                "demo_sudoku_2.jpg"
            ]
            
            # Check if any sample files exist
            for sample in sample_files:
                if os.path.exists(sample):
                    print(f"Using sample file: {sample}")
                    return sample
            
            # Check if there are any image files in the directory
            import os
            import glob
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(glob.glob(os.path.join(os.path.dirname(__file__), ext)))
            
            if image_files:
                print(f"Found {len(image_files)} image files: {[os.path.basename(f) for f in image_files]}")
                # Use the first image file found (or you could cycle through them)
                return image_files[0]
            
            # Try to use system file dialog without tkinter
            try:
                if platform.system() == "Windows":
                    # Use PowerShell file dialog on Windows
                    ps_command = '''
                    Add-Type -AssemblyName System.Windows.Forms
                    $dialog = New-Object System.Windows.Forms.OpenFileDialog
                    $dialog.Filter = "Image files (*.jpg;*.jpeg;*.png;*.bmp)|*.jpg;*.jpeg;*.png;*.bmp|All files (*.*)|*.*"
                    $dialog.Title = "Select Sudoku Image"
                    $dialog.InitialDirectory = [System.IO.Path]::GetDirectoryName($MyInvocation.MyCommand.Path)
                    $result = $dialog.ShowDialog()
                    if ($result -eq "OK") {
                        Write-Output $dialog.FileName
                    }
                    '''
                    
                    result = subprocess.run(
                        ["powershell", "-Command", ps_command],
                        capture_output=True,
                        text=True,
                        timeout=60,  # Increased timeout
                        cwd=os.path.dirname(__file__)  # Set working directory
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        selected_file = result.stdout.strip()
                        print(f"User selected file: {selected_file}")
                        return selected_file
                    else:
                        print(f"File dialog cancelled or failed")
                
            except Exception as e:
                print(f"System dialog failed: {e}")
            
            # Fallback: Create a demo image and use it
            print("Creating demo image for testing...")
            demo_path = self.create_demo_image()
            if demo_path:
                print(f"Using created demo image: {demo_path}")
                return demo_path
            
            print("No image selected or created")
            return None
            
        except Exception as e:
            print(f"Error in file selection: {e}")
            # Create demo image as last resort
            try:
                demo_path = self.create_demo_image()
                if demo_path:
                    return demo_path
            except:
                pass
            return None
    
    def create_demo_image(self) -> Optional[str]:
        """
        Create a demo Sudoku image for testing when file selection fails.
        
        Returns:
            Path to created demo image or None if failed
        """
        try:
            import numpy as np
            
            # Try to import opencv, if not available, create simple demo data
            try:
                import cv2
                HAS_CV2 = True
            except ImportError:
                HAS_CV2 = False
            
            if HAS_CV2:
                # Create a simple Sudoku grid image
                img_size = 450
                img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
                
                # Sample puzzle data
                puzzle_data = [
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
                
                # Draw grid lines
                cell_size = img_size // 9
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
                            x = col * cell_size + cell_size // 2 - 10
                            y = row * cell_size + cell_size // 2 + 10
                            cv2.putText(img, str(puzzle_data[row][col]), (x, y), 
                                       font, font_scale, (0, 0, 0), font_thickness)
                
                # Save demo image
                demo_filename = "demo_sudoku_auto.jpg"
                cv2.imwrite(demo_filename, img)
                print(f"Created demo image: {demo_filename}")
                return demo_filename
            
            else:
                # Create a text file indicating the demo puzzle for testing
                demo_filename = "demo_puzzle_data.txt"
                puzzle_text = """Demo Sudoku Puzzle:
5 3 0 0 7 0 0 0 0
6 0 0 1 9 5 0 0 0
0 9 8 0 0 0 0 6 0
8 0 0 0 6 0 0 0 3
4 0 0 8 0 3 0 0 1
7 0 0 0 2 0 0 0 6
0 6 0 0 0 0 2 8 0
0 0 0 4 1 9 0 0 5
0 0 0 0 8 0 0 7 9"""
                
                with open(demo_filename, 'w') as f:
                    f.write(puzzle_text)
                
                print(f"Created demo puzzle data: {demo_filename}")
                return demo_filename
        
        except Exception as e:
            print(f"Failed to create demo image: {e}")
            return None
    
    def process_image_async(self, image_path: str, algorithm: str = "backtrack"):
        """
        Process image in a separate thread to avoid blocking the GUI.
        
        Args:
            image_path: Path to the image file
            algorithm: Solving algorithm to use
        """
        def processing_worker():
            try:
                self.processing_queue.put(("status", "Loading image..."))
                self.processing_queue.put(("progress", 0.1))
                
                # Check if this is a demo text file
                if image_path.endswith('.txt'):
                    self.processing_queue.put(("status", "Loading demo puzzle..."))
                    self.processing_queue.put(("progress", 0.5))
                    
                    # Load demo puzzle from text file
                    try:
                        with open(image_path, 'r') as f:
                            content = f.read()
                        
                        # Extract puzzle from text (simple format)
                        lines = content.strip().split('\n')
                        puzzle_lines = [line for line in lines if any(c.isdigit() for c in line)]
                        
                        sample_puzzle = []
                        for line in puzzle_lines:
                            row = []
                            for char in line:
                                if char.isdigit():
                                    row.append(int(char))
                                elif char == '0' or char == ' ' or char == '.':
                                    if len(row) < 9:
                                        row.append(0)
                            if len(row) == 9:
                                sample_puzzle.append(row)
                                if len(sample_puzzle) == 9:
                                    break
                        
                        # Fill remaining with demo data if needed
                        if len(sample_puzzle) < 9:
                            sample_puzzle = [
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
                    
                    except Exception as e:
                        print(f"Error loading demo: {e}")
                        # Default demo puzzle
                        sample_puzzle = [
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
                
                else:
                    # Process actual image using enhanced mobile processor
                    try:
                        # Try the enhanced mobile processor first
                        from mobile_image_processor import MobileSudokuProcessor
                        
                        # Initialize the mobile processor
                        mobile_processor = MobileSudokuProcessor()
                        
                        self.processing_queue.put(("status", "Analyzing mobile screenshot..."))
                        self.processing_queue.put(("progress", 0.3))
                        
                        # Process the actual image with mobile-optimized algorithms
                        print(f"Processing mobile image: {image_path}")
                        extracted_puzzle = mobile_processor.process_mobile_image(image_path)
                        
                        if extracted_puzzle:
                            self.processing_queue.put(("status", "Successfully extracted from mobile screenshot!"))
                            self.processing_queue.put(("progress", 0.7))
                            
                            # Use the extracted puzzle
                            sample_puzzle = extracted_puzzle
                            print(f"Extracted puzzle from mobile screenshot {image_path}:")
                            for row in sample_puzzle:
                                print(row)
                            
                            self.already_solved_board = None  # Let the regular solver handle it
                                
                        else:
                            # If mobile extraction failed, try the original processor
                            print(f"Mobile extraction failed, trying original processor...")
                            
                            try:
                                from image_solver import SudokuImageProcessor
                                processor = SudokuImageProcessor()
                                
                                self.processing_queue.put(("status", "Trying standard image processing..."))
                                self.processing_queue.put(("progress", 0.5))
                                
                                result = processor.solve_from_image(image_path)
                                
                                if result:
                                    extracted_board, solved_board = result
                                    self.processing_queue.put(("status", "Successfully processed with standard method!"))
                                    
                                    # Convert Board objects to matrix format for display
                                    sample_puzzle = []
                                    for row in range(9):
                                        puzzle_row = []
                                        for col in range(9):
                                            puzzle_row.append(extracted_board.get_cell(row, col))
                                        sample_puzzle.append(puzzle_row)
                                    
                                    self.already_solved_board = solved_board
                                else:
                                    raise Exception("Standard processor also failed")
                                    
                            except Exception as e2:
                                # Both methods failed, show error but continue with demo
                                print(f"Both image processing methods failed: {e2}")
                                self.processing_queue.put(("status", f"Could not extract from image. Using demo puzzle."))
                                self.processing_queue.put(("progress", 0.5))
                                
                                # Fall back to demo puzzle
                                sample_puzzle = [
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
                                self.already_solved_board = None
                        
                    except ImportError as e:
                        print(f"Mobile processor not available: {e}")
                        self.processing_queue.put(("status", "Mobile processor unavailable. Using demo puzzle."))
                        self.processing_queue.put(("progress", 0.5))
                        
                        # Use demo puzzle
                        sample_puzzle = [
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
                        self.already_solved_board = None
                        
                    except Exception as e:
                        print(f"Image processing failed: {e}")
                        self.processing_queue.put(("error", f"Image processing failed: {str(e)}"))
                        return
                
                # Create boards from the puzzle data
                extracted_board = Board()
                for row in range(9):
                    for col in range(9):
                        if sample_puzzle[row][col] != 0:
                            extracted_board.set_cell(row, col, sample_puzzle[row][col])
                
                # Check if we already have a solved board from image processor
                if hasattr(self, 'already_solved_board') and self.already_solved_board:
                    self.processing_queue.put(("status", f"Using solution from image processor!"))
                    self.processing_queue.put(("progress", 1.0))
                    solved_board = self.already_solved_board
                    success = True
                else:
                    self.processing_queue.put(("status", "Solving puzzle..."))
                    self.processing_queue.put(("progress", 0.9))
                    
                    # Solve the puzzle
                    solved_board = Board()
                    for row in range(9):
                        for col in range(9):
                            solved_board.set_cell(row, col, extracted_board.get_cell(row, col))
                    
                    # Choose solver based on algorithm and handle return values properly
                    if algorithm == "dlx":
                        # DLX solver returns (success, solution_board)
                        solver = self.dlx_solver
                        success, solution_board = solver.solve(solved_board)
                        if success:
                            solved_board = solution_board
                    else:
                        # Backtracking solver returns (success, solution_board)
                        solver = self.backtrack_solver
                        success, solution_board = solver.solve(solved_board)
                        if success:
                            solved_board = solution_board
                
                if success:
                    self.processing_queue.put(("result", (extracted_board, solved_board)))
                    self.processing_queue.put(("status", f"Solved using {algorithm.upper()}!"))
                else:
                    self.processing_queue.put(("error", f"Failed to solve puzzle with {algorithm}"))
                
                self.processing_queue.put(("progress", 1.0))
                
            except Exception as e:
                self.processing_queue.put(("error", f"Processing failed: {str(e)}"))
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def update_processing_status(self):
        """Update processing status from the background thread."""
        try:
            while not self.processing_queue.empty():
                message_type, data = self.processing_queue.get_nowait()
                
                if message_type == "status":
                    self.processing_status = data
                elif message_type == "progress":
                    self.processing_progress = data
                elif message_type == "result":
                    self.extracted_board, self.solved_board = data
                    self.processing_error = None
                elif message_type == "error":
                    self.processing_error = data
                    self.processing_status = "Error"
                    
        except queue.Empty:
            pass
    
    def draw_image_interface(self, y_offset: int = 100) -> int:
        """
        Draw the image processing interface.
        
        Args:
            y_offset: Y position to start drawing
            
        Returns:
            Height of the drawn interface
        """
        if not self.show_image_interface:
            return 0
        
        start_y = y_offset
        current_y = start_y
        margin = 15
        
        # Title
        title_text = self.font.render("Image Sudoku Solver", True, BLACK)
        title_rect = title_text.get_rect(centerx=self.screen.get_width() // 2, y=current_y)
        self.screen.blit(title_text, title_rect)
        current_y += title_text.get_height() + margin
        
        # Current image path
        if self.current_image_path:
            path_text = f"Image: {os.path.basename(self.current_image_path)}"
            if len(path_text) > 60:
                path_text = path_text[:57] + "..."
        else:
            path_text = "No image selected"
        
        path_surface = self.small_font.render(path_text, True, GRAY)
        path_rect = path_surface.get_rect(centerx=self.screen.get_width() // 2, y=current_y)
        self.screen.blit(path_surface, path_rect)
        current_y += path_surface.get_height() + margin
        
        # Buttons with better spacing
        button_width = 180
        button_height = 35
        button_spacing = 25
        
        buttons = [
            ("Load Image", "load_image"),
            ("Process Image", "process_image"),
            ("Algorithm", "toggle_algorithm"),
        ]
        
        # Calculate total width for centering
        total_width = len(buttons) * button_width + (len(buttons) - 1) * button_spacing
        start_x = (self.screen.get_width() - total_width) // 2
        
        self.button_rects = {}  # Store button rectangles for click detection
        
        for i, (text, action) in enumerate(buttons):
            x = start_x + i * (button_width + button_spacing)
            rect = pygame.Rect(x, current_y, button_width, button_height)
            
            # Button state
            enabled = True
            if action == "process_image" and not self.current_image_path:
                enabled = False
            
            # Draw button
            color = LIGHT_BLUE if enabled else LIGHT_GRAY
            if enabled and rect.collidepoint(pygame.mouse.get_pos()):
                color = BLUE
            
            pygame.draw.rect(self.screen, color, rect)
            pygame.draw.rect(self.screen, BLACK, rect, 2)
            
            # Button text
            text_color = BLACK if enabled else GRAY
            if action == "toggle_algorithm":
                display_text = f"{self.selected_algorithm.upper()}"
            else:
                display_text = text
            
            text_surface = self.small_font.render(display_text, True, text_color)
            text_rect = text_surface.get_rect(center=rect.center)
            self.screen.blit(text_surface, text_rect)
            
            self.button_rects[action] = rect
        
        current_y += button_height + margin * 2
        
        # Processing status with better layout
        if hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive():
            # Progress bar
            progress_width = 350
            progress_height = 20
            progress_rect = pygame.Rect(
                (self.screen.get_width() - progress_width) // 2,
                current_y,
                progress_width,
                progress_height
            )
            
            # Progress background
            pygame.draw.rect(self.screen, LIGHT_GRAY, progress_rect)
            pygame.draw.rect(self.screen, BLACK, progress_rect, 2)
            
            # Progress fill
            fill_width = int(progress_width * self.processing_progress)
            if fill_width > 0:
                fill_rect = pygame.Rect(progress_rect.x, progress_rect.y, fill_width, progress_height)
                pygame.draw.rect(self.screen, GREEN, fill_rect)
            
            current_y += progress_height + 10
            
            # Status text
            status_surface = self.small_font.render(self.processing_status, True, BLACK)
            status_rect = status_surface.get_rect(centerx=self.screen.get_width() // 2, y=current_y)
            self.screen.blit(status_surface, status_rect)
            current_y += status_surface.get_height() + margin
        
        # Error display
        if self.processing_error:
            error_surface = self.small_font.render(f"Error: {self.processing_error}", True, RED)
            error_rect = error_surface.get_rect(centerx=self.screen.get_width() // 2, y=current_y)
            self.screen.blit(error_surface, error_rect)
            current_y += error_surface.get_height() + margin
        
        # Results display with better positioning
        if self.extracted_board and self.solved_board:
            # Draw side-by-side boards with proper spacing
            board_size = 180
            board_spacing = 60
            
            # Calculate positions
            total_board_width = 2 * board_size + board_spacing
            board_start_x = (self.screen.get_width() - total_board_width) // 2
            
            # Labels with better positioning
            extracted_label = self.small_font.render("Extracted Puzzle", True, BLACK)
            solved_label = self.small_font.render("Solution", True, BLACK)
            
            extracted_rect = extracted_label.get_rect(centerx=board_start_x + board_size // 2, y=current_y)
            solved_rect = solved_label.get_rect(centerx=board_start_x + board_size + board_spacing + board_size // 2, y=current_y)
            
            self.screen.blit(extracted_label, extracted_rect)
            self.screen.blit(solved_label, solved_rect)
            current_y += extracted_label.get_height() + 10
            
            # Draw boards with borders
            board_y = current_y
            
            # Draw background for boards
            pygame.draw.rect(self.screen, WHITE, (board_start_x - 5, board_y - 5, board_size + 10, board_size + 10))
            pygame.draw.rect(self.screen, BLACK, (board_start_x - 5, board_y - 5, board_size + 10, board_size + 10), 2)
            
            pygame.draw.rect(self.screen, WHITE, (board_start_x + board_size + board_spacing - 5, board_y - 5, board_size + 10, board_size + 10))
            pygame.draw.rect(self.screen, BLACK, (board_start_x + board_size + board_spacing - 5, board_y - 5, board_size + 10, board_size + 10), 2)
            
            # Draw the actual boards
            self.draw_mini_board(self.extracted_board, board_start_x, board_y, board_size)
            self.draw_mini_board(self.solved_board, board_start_x + board_size + board_spacing, board_y, board_size)
            current_y += board_size + margin * 2
        
        return current_y - start_y
    
    def draw_mini_board(self, board: Board, x: int, y: int, size: int):
        """
        Draw a miniature version of the Sudoku board.
        
        Args:
            board: Board to draw
            x, y: Top-left position
            size: Size of the board
        """
        cell_size = size // 9
        
        # Draw white background
        pygame.draw.rect(self.screen, WHITE, (x, y, size, size))
        
        # Draw grid lines
        for i in range(10):
            thickness = 3 if i % 3 == 0 else 1
            color = BLACK
            
            # Vertical lines
            start_pos = (x + i * cell_size, y)
            end_pos = (x + i * cell_size, y + size)
            pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
            
            # Horizontal lines
            start_pos = (x, y + i * cell_size)
            end_pos = (x + size, y + i * cell_size)
            pygame.draw.line(self.screen, color, start_pos, end_pos, thickness)
        
        # Draw numbers with better sizing and positioning
        font_size = max(12, min(24, cell_size - 8))
        number_font = pygame.font.Font(None, font_size)
        
        for row in range(9):
            for col in range(9):
                value = board.get_cell(row, col)
                if value != 0:
                    # Calculate center position more precisely
                    center_x = x + col * cell_size + cell_size // 2
                    center_y = y + row * cell_size + cell_size // 2
                    
                    text_surface = number_font.render(str(value), True, BLACK)
                    text_rect = text_surface.get_rect(center=(center_x, center_y))
                    self.screen.blit(text_surface, text_rect)
    
    def handle_image_interface_click(self, pos: Tuple[int, int]) -> bool:
        """
        Handle mouse clicks in the image interface.
        
        Args:
            pos: Mouse click position
            
        Returns:
            True if click was handled
        """
        if not self.show_image_interface:
            return False
        
        # Update processing status first
        self.update_processing_status()
        
        # Check if we have stored button rectangles
        if hasattr(self, 'button_rects'):
            mouse_x, mouse_y = pos
            
            # Check each button
            for action, rect in self.button_rects.items():
                if rect.collidepoint(mouse_x, mouse_y):
                    if action == "load_image":
                        image_path = self.select_image_file()
                        if image_path:
                            self.current_image_path = image_path
                            self.reset_processing_state()
                            self.current_image_path = image_path  # Keep the selected path
                        return True
                    
                    elif action == "process_image" and self.current_image_path:
                        if not (hasattr(self, 'processing_thread') and self.processing_thread and self.processing_thread.is_alive()):
                            self.process_image_async(self.current_image_path, self.selected_algorithm)
                        return True
                    
                    elif action == "toggle_algorithm":
                        self.selected_algorithm = "dlx" if self.selected_algorithm == "backtrack" else "backtrack"
                        return True
        
        return False


def create_image_solver_demo():
    """
    Create a standalone demo of the image solver GUI integration.
    """
    pygame.init()
    
    # Screen setup
    screen_width, screen_height = 1000, 800
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Sudoku Image Solver Demo")
    
    # Fonts
    font = pygame.font.Font(None, 36)
    
    # Create image solver GUI
    image_gui = ImageSolverGUI(screen, font)
    image_gui.show_image_interface = True
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    image_gui.handle_image_interface_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_i:
                    image_gui.toggle_image_interface()
        
        # Update processing status
        image_gui.update_processing_status()
        
        # Draw everything
        screen.fill(WHITE)
        
        # Title
        title = font.render("Sudoku Image Solver Demo", True, BLACK)
        title_rect = title.get_rect(centerx=screen_width // 2, y=30)
        screen.blit(title, title_rect)
        
        # Instructions
        instructions = [
            "Press 'I' to toggle image interface",
            "Click 'Load Image' to select a Sudoku photo",
            "Click 'Process Image' to extract and solve",
            "ESC to exit"
        ]
        
        y_pos = 60
        for instruction in instructions:
            text = pygame.font.Font(None, 24).render(instruction, True, GRAY)
            text_rect = text.get_rect(centerx=screen_width // 2, y=y_pos)
            screen.blit(text, text_rect)
            y_pos += 25
        
        # Draw image interface
        interface_height = image_gui.draw_image_interface(150)
        
        # Status note
        note_y = 150 + interface_height + 20
        note_text = "Note: Full functionality requires OpenCV and EasyOCR installation"
        note_surface = pygame.font.Font(None, 20).render(note_text, True, RED)
        note_rect = note_surface.get_rect(centerx=screen_width // 2, y=note_y)
        screen.blit(note_surface, note_rect)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()


if __name__ == "__main__":
    create_image_solver_demo()
