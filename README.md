# 🧩 Sudoku Solver & Computer Vision Analyzer

A comprehensive Sudoku solving application with advanced computer vision capabilities for mobile game screenshot processing.

![Sudoku Solver Demo](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-GUI-red.svg)

## 🌟 Features

### 🎯 **Dual Algorithm Support**
- **Backtracking Algorithm**: Classic recursive solver with optimization
- **Dancing Links (DLX)**: Advanced exact cover algorithm by Donald Knuth
- **Performance Comparison**: Real-time benchmarking and analysis

### 🖼️ **Computer Vision Integration**
- **Mobile Screenshot Processing**: Extract Sudoku puzzles from mobile game screenshots
- **Advanced OCR**: EasyOCR integration for accurate digit recognition
- **Multiple Detection Strategies**: Handles various image qualities and formats
- **Visual File Browser**: Interactive image selection interface

### 🎮 **Interactive GUI**
- **Real-time Visualization**: Watch algorithms solve puzzles step-by-step
- **Manual Input**: Click-to-place digits for custom puzzles
- **Performance Metrics**: Live algorithm comparison and timing
- **Image Processing Interface**: Load, process, and solve from screenshots

### 🔧 **Advanced Features**
- **Puzzle Generation**: Create puzzles with multiple difficulty levels
- **Validation System**: Ensure puzzle correctness and uniqueness
- **Performance Analytics**: Export timing and efficiency data
- **Comprehensive Testing**: Full test suite for all components

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sudoku-solver-cv.git
cd sudoku-solver-cv

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

### Dependencies

- **Python 3.8+**
- **OpenCV** (`opencv-python`) - Computer vision operations
- **EasyOCR** (`easyocr`) - Optical character recognition
- **Pygame** (`pygame`) - GUI and visualization
- **NumPy** (`numpy`) - Numerical computations
- **Pillow** (`pillow`) - Image processing utilities

## 📱 Using the Image Processing Feature

### 1. **Load Image**
- Click "Load Image" button
- Use the visual browser to select a Sudoku screenshot
- Supports JPG, PNG, and BMP formats

### 2. **Process Image**
- Click "Process Image" to extract the puzzle
- The system automatically detects the grid and recognizes digits
- View the extracted puzzle in the "Extracted Puzzle" section

### 3. **Solve**
- Choose between "BACKTRACK" or "DLX" algorithms
- Watch the real-time solving process
- Compare algorithm performance

## 🧠 Algorithm Comparison

| Algorithm | Strengths | Best For |
|-----------|-----------|----------|
| **Backtracking** | Simple, intuitive, good for learning | Easy to medium puzzles |
| **Dancing Links** | Extremely fast, memory efficient | Hard puzzles, performance critical |

## 📁 Project Structure

```
sudoku-solver-cv/
├── main.py                     # Main application entry point
├── board.py                    # Sudoku board logic
├── solver_backtrack.py         # Backtracking algorithm
├── solver_dlx.py              # Dancing Links algorithm
├── generator.py               # Puzzle generator
├── performance.py             # Performance analysis
├── visualizer.py              # GUI interface
├── mobile_image_processor.py  # Image processing pipeline
├── image_gui_integration.py   # GUI integration
├── image_browser.py           # File browser component
├── requirements.txt           # Dependencies
└── tests/                     # Test suite
```

## 🖼️ Image Processing Pipeline

### Detection Strategies
1. **Clear Image Grid Detection**: Optimized for high-quality screenshots
2. **Contour Detection**: Handles various backgrounds and lighting
3. **Adaptive Thresholding**: Robust to different image conditions
4. **Full-Image Fallback**: Ensures compatibility with edge cases

### OCR Enhancement
- **Multiple Preprocessing Methods**: Binary, adaptive, and Otsu thresholding
- **Confidence-Based Validation**: Ensures accurate digit recognition
- **Error Correction**: Handles common OCR mistakes (O→0, I→1, etc.)
- **Mobile Optimization**: Specifically tuned for mobile game interfaces

## 🎯 Examples

### Manual Solving
```python
from board import SudokuBoard
from solver_backtrack import BacktrackSolver

# Create a board and solver
board = SudokuBoard()
solver = BacktrackSolver()

# Load a puzzle
puzzle = [[5,3,0,0,7,0,0,0,0],
          [6,0,0,1,9,5,0,0,0],
          # ... more rows
          ]
board.load_puzzle(puzzle)

# Solve
solution = solver.solve(board)
```

### Image Processing
```python
from mobile_image_processor import MobileSudokuProcessor

# Process a mobile screenshot
processor = MobileSudokuProcessor()
puzzle_matrix = processor.process_mobile_sudoku('screenshot.jpg')

# The extracted puzzle is ready for solving
if puzzle_matrix:
    print("Successfully extracted puzzle!")
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_backtrack.py    # Backtracking tests
python -m pytest tests/test_dlx.py          # DLX tests
python -m pytest tests/test_image_solver.py # Image processing tests
```

## 📊 Performance Benchmarks

The application includes built-in performance analysis:

- **Algorithm Timing**: Microsecond precision timing
- **Memory Usage**: Track memory consumption patterns
- **Success Rates**: Monitor solving accuracy
- **Comparison Reports**: Side-by-side algorithm analysis

Results are exported to JSON format for further analysis.

## 🔧 Configuration

### Image Processing Settings
- **OCR Confidence Thresholds**: Adjust for accuracy vs coverage
- **Grid Detection Sensitivity**: Fine-tune for different image types
- **Preprocessing Parameters**: Customize for specific mobile games

### Solver Options
- **Visualization Speed**: Control step-by-step display timing
- **Algorithm Selection**: Choose default solving method
- **Performance Tracking**: Enable/disable detailed metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Donald Knuth** - Dancing Links algorithm
- **EasyOCR Team** - Optical character recognition
- **OpenCV Community** - Computer vision tools
- **Pygame Developers** - GUI framework

## 🐛 Known Issues & Future Improvements

### Current Limitations
- OCR accuracy varies with image quality
- Processing time depends on image complexity
- Limited to standard 9x9 Sudoku grids

### Planned Features
- [ ] Support for different grid sizes (4x4, 16x16)
- [ ] Multiple language OCR support
- [ ] Advanced image preprocessing filters
- [ ] Machine learning digit recognition
- [ ] Web interface version
- [ ] Mobile app development

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/yourusername/sudoku-solver-cv/issues) page
2. Create a new issue with detailed description
3. Include sample images for image processing problems
4. Provide system information and error logs

---

⭐ **If you find this project helpful, please give it a star!** ⭐
