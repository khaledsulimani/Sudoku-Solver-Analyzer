# 🧩 Sudoku Solver & Computer Vision Analyzer

A comprehensive Sudoku solving application with advanced computer vision capabilities for mobile game screenshot processing.

![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green.svg)
![Pygame](https://img.shields.io/badge/Pygame-GUI-red.svg)
![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-complete-brightgreen.svg)


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

## 📸 Project Results

### *python app result*:  


![صورة واتساب بتاريخ 1447-02-25 في 08 44 46_37a286f5](https://github.com/user-attachments/assets/c08d392b-938c-43c3-b7a6-3e131ae3467f)




### *the phone app after solving soduku with python app*:

![صورة واتساب بتاريخ 1447-02-25 في 08 45 55_cd31f9d1](https://github.com/user-attachments/assets/9f370e80-d0e9-4265-9457-7faab0d4b34a)




### *gui main*:
<img width="1799" height="1243" alt="image" src="https://github.com/user-attachments/assets/cafc7ae6-3477-4527-a442-c99b0b787b94" />

### *gui hard level genrated soduku*:
<img width="1799" height="1241" alt="image" src="https://github.com/user-attachments/assets/7bfaca92-db0e-41a2-80ec-ee35b468f469" />

### *gui hard level interface solved by algorithm*:
<img width="1794" height="1240" alt="image" src="https://github.com/user-attachments/assets/0307af0a-a286-49a3-a1ab-5649eb79ec88" />

### *gui performance analysis*:
<img width="1798" height="1243" alt="image" src="https://github.com/user-attachments/assets/1dec692f-e770-4192-8a44-a6ff9cf18cee" />

### *gui image solver*:
<img width="1799" height="1240" alt="image" src="https://github.com/user-attachments/assets/4ad097f7-1407-4b45-b3a8-adfa0972d7f3" />



### *graph for Algorithm*:

## Execution Time Comparison:
<img width="1503" height="991" alt="image" src="https://github.com/user-attachments/assets/ade16031-5aaf-4d12-a694-ae45db4f373a" />


## Steps Taken Comparison:
<img width="1496" height="986" alt="image" src="https://github.com/user-attachments/assets/2de1d44c-7425-4738-8a7a-2481e09fcc57" />

## Memory Usage Comparison:
<img width="1503" height="989" alt="image" src="https://github.com/user-attachments/assets/e1a4b8a7-5198-44a3-8574-3c59dd62634a" />

## Comprehensive Dashboard:
<img width="2260" height="1435" alt="image" src="https://github.com/user-attachments/assets/ebd51ea7-008d-4a63-a51e-f3ad32416a3a" />

## Historical Performance Trends:
<img width="1801" height="992" alt="image" src="https://github.com/user-attachments/assets/a924c8f1-5ddc-46ad-8825-9ab19e044b46" />

## exit graph:
<img width="1804" height="1288" alt="image" src="https://github.com/user-attachments/assets/1ed3ea9a-6ad5-4312-929c-76f30fedeba2" />

---

---
  
## 🧑‍💻 Author
- **khaled mahmoud sulaimani** – [@khaledsulimani](https://github.com/khaledsulimani)

---


⭐ **If you find this project helpful, please give it a star!** ⭐
