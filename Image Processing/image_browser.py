"""
Image File Browser for Sudoku Solver
====================================
Allows browsing and selecting different Sudoku images.
"""

import os
import pygame
from typing import List, Optional

class ImageFileBrowser:
    """Simple file browser for selecting Sudoku images."""
    
    def __init__(self, screen: pygame.Surface, font: pygame.font.Font):
        self.screen = screen
        self.font = font
        self.small_font = pygame.font.Font(None, 20)
        self.current_directory = os.path.dirname(__file__)
        self.image_files = self.scan_for_images()
        self.selected_index = 0
        self.scroll_offset = 0
        
    def scan_for_images(self) -> List[str]:
        """Scan current directory for image files."""
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        image_files = []
        
        try:
            for file in os.listdir(self.current_directory):
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(file)
            image_files.sort()
        except Exception as e:
            print(f"Error scanning directory: {e}")
            
        return image_files
    
    def draw(self, x: int, y: int, width: int, height: int):
        """Draw the file browser interface."""
        # Draw background
        browser_rect = pygame.Rect(x, y, width, height)
        pygame.draw.rect(self.screen, (240, 240, 240), browser_rect)
        pygame.draw.rect(self.screen, (0, 0, 0), browser_rect, 2)
        
        # Draw title
        title_text = self.font.render("Select Sudoku Image:", True, (0, 0, 0))
        self.screen.blit(title_text, (x + 10, y + 10))
        
        # Draw file list
        list_y = y + 40
        list_height = height - 80
        item_height = 30
        visible_items = list_height // item_height
        
        if not self.image_files:
            no_files_text = self.small_font.render("No image files found in directory", True, (150, 150, 150))
            self.screen.blit(no_files_text, (x + 10, list_y + 10))
            return None
        
        # Draw files
        for i in range(min(visible_items, len(self.image_files))):
            file_index = i + self.scroll_offset
            if file_index >= len(self.image_files):
                break
                
            file_name = self.image_files[file_index]
            item_y = list_y + i * item_height
            item_rect = pygame.Rect(x + 5, item_y, width - 10, item_height - 2)
            
            # Highlight selected item
            if file_index == self.selected_index:
                pygame.draw.rect(self.screen, (100, 150, 255), item_rect)
                text_color = (255, 255, 255)
            else:
                pygame.draw.rect(self.screen, (255, 255, 255), item_rect)
                text_color = (0, 0, 0)
            
            pygame.draw.rect(self.screen, (0, 0, 0), item_rect, 1)
            
            # Draw file name
            text = self.small_font.render(file_name, True, text_color)
            self.screen.blit(text, (item_rect.x + 5, item_rect.y + 5))
        
        # Draw buttons
        button_y = y + height - 35
        
        # Select button
        select_button = pygame.Rect(x + 10, button_y, 80, 25)
        pygame.draw.rect(self.screen, (100, 200, 100), select_button)
        pygame.draw.rect(self.screen, (0, 0, 0), select_button, 2)
        select_text = self.small_font.render("Select", True, (0, 0, 0))
        self.screen.blit(select_text, (select_button.x + 15, select_button.y + 5))
        
        # Refresh button
        refresh_button = pygame.Rect(x + 100, button_y, 80, 25)
        pygame.draw.rect(self.screen, (200, 200, 100), refresh_button)
        pygame.draw.rect(self.screen, (0, 0, 0), refresh_button, 2)
        refresh_text = self.small_font.render("Refresh", True, (0, 0, 0))
        self.screen.blit(refresh_text, (refresh_button.x + 10, refresh_button.y + 5))
        
        # Cancel button
        cancel_button = pygame.Rect(x + 190, button_y, 80, 25)
        pygame.draw.rect(self.screen, (200, 100, 100), cancel_button)
        pygame.draw.rect(self.screen, (0, 0, 0), cancel_button, 2)
        cancel_text = self.small_font.render("Cancel", True, (0, 0, 0))
        self.screen.blit(cancel_text, (cancel_button.x + 15, cancel_button.y + 5))
        
        return {
            'select': select_button,
            'refresh': refresh_button,
            'cancel': cancel_button,
            'list_area': pygame.Rect(x + 5, list_y, width - 10, list_height)
        }
    
    def handle_click(self, pos: tuple, button_rects: dict) -> Optional[str]:
        """Handle mouse clicks in the browser."""
        if not button_rects:
            return None
            
        # Check button clicks
        if button_rects['select'].collidepoint(pos):
            if self.image_files and 0 <= self.selected_index < len(self.image_files):
                selected_file = os.path.join(self.current_directory, self.image_files[self.selected_index])
                return selected_file
                
        elif button_rects['refresh'].collidepoint(pos):
            self.image_files = self.scan_for_images()
            self.selected_index = 0
            self.scroll_offset = 0
            
        elif button_rects['cancel'].collidepoint(pos):
            return "CANCEL"
            
        elif button_rects['list_area'].collidepoint(pos):
            # Click in file list
            list_y = button_rects['list_area'].y
            item_height = 30
            clicked_item = (pos[1] - list_y) // item_height
            file_index = clicked_item + self.scroll_offset
            
            if 0 <= file_index < len(self.image_files):
                self.selected_index = file_index
        
        return None
    
    def handle_key(self, key: int):
        """Handle keyboard input."""
        if key == pygame.K_UP:
            self.selected_index = max(0, self.selected_index - 1)
        elif key == pygame.K_DOWN:
            self.selected_index = min(len(self.image_files) - 1, self.selected_index + 1)
        elif key == pygame.K_RETURN:
            if self.image_files and 0 <= self.selected_index < len(self.image_files):
                return os.path.join(self.current_directory, self.image_files[self.selected_index])
        
        return None


def show_image_browser(screen: pygame.Surface, font: pygame.font.Font) -> Optional[str]:
    """
    Show the image browser and return selected file path.
    
    Returns:
        Selected file path, "CANCEL", or None
    """
    browser = ImageFileBrowser(screen, font)
    clock = pygame.time.Clock()
    
    # Position the browser in the center of the screen
    browser_width = 400
    browser_height = 500
    browser_x = (screen.get_width() - browser_width) // 2
    browser_y = (screen.get_height() - browser_height) // 2
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "CANCEL"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    button_rects = browser.draw(browser_x, browser_y, browser_width, browser_height)
                    result = browser.handle_click(event.pos, button_rects)
                    if result:
                        return result
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return "CANCEL"
                else:
                    result = browser.handle_key(event.key)
                    if result:
                        return result
        
        # Clear screen
        screen.fill((50, 50, 50))
        
        # Draw browser
        button_rects = browser.draw(browser_x, browser_y, browser_width, browser_height)
        
        # Draw instructions
        instructions = [
            "Use UP/DOWN arrows or click to select file",
            "Press ENTER or click Select to choose",
            "Press ESC or click Cancel to exit"
        ]
        
        for i, instruction in enumerate(instructions):
            text = pygame.font.Font(None, 24).render(instruction, True, (255, 255, 255))
            text_x = (screen.get_width() - text.get_width()) // 2
            text_y = browser_y + browser_height + 20 + i * 25
            screen.blit(text, (text_x, text_y))
        
        pygame.display.flip()
        clock.tick(60)
    
    return None


if __name__ == "__main__":
    # Test the image browser
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Image Browser Test")
    font = pygame.font.Font(None, 36)
    
    result = show_image_browser(screen, font)
    print(f"Selected: {result}")
    
    pygame.quit()
