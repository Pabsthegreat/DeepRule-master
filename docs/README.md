# DeepRule Documentation

Welcome to the comprehensive documentation for DeepRule - a deep learning-based system for chart analysis and data extraction.

## 📚 Documentation Structure

### Getting Started
- [Installation Guide](installation.md) - Setup instructions and dependencies
- [Quick Start](quick-start.md) - Run your first chart extraction

### Core Concepts
- [Architecture Overview](architecture.md) - System design and components
- [Model Details](models.md) - Neural network architectures explained
- [Dataset Format](dataset.md) - Training data structure and annotations

### Training
- [Training Guide](training.md) - How to train models from scratch
- [Configuration Files](configuration.md) - Understanding config parameters

### Inference
- [Inference Guide](inference.md) - Using trained models for prediction
- [API Reference](api-reference.md) - Web server and programmatic API
- [Pipeline Explanation](pipeline.md) - End-to-end processing flow

### Advanced Topics
- [Chart Type Detection](chart-detection.md) - Auto-detection algorithm
- [OCR Integration](ocr.md) - Text extraction and processing
- [X-axis Label Extraction](xaxis-extraction.md) - Handling rotated/tilted text
- [Rule-Based Processing](rules.md) - Post-processing rules

### Maintenance
- [Code Structure](code-structure.md) - Repository organization
- [Troubleshooting](troubleshooting.md) - Common issues and solutions

## 🎯 What is DeepRule?

DeepRule is an end-to-end system that:
1. **Detects chart types** (Bar, Line, Pie)
2. **Extracts visual elements** (bars, lines, pie segments, axes, legends)
3. **Performs OCR** on text elements (labels, values, titles)
4. **Applies rules** to structure and validate extracted data
5. **Outputs structured data** (CSV format with values and labels)

## 🏗️ System Architecture

```
Input Image → Chart Type Detection → Element Detection → OCR → Rule Processing → Structured Output
     ↓              ↓                      ↓              ↓           ↓              ↓
  chart.png    Bar/Line/Pie         Bounding boxes    Text data   Validation     CSV file
```

## 🚀 Quick Example

```bash
# Start web server
python manage.py runserver 0.0.0.0:8000

# Or run batch processing
python test_pipe_type_cloud.py \
  --image_path sample.png \
  --save_path output \
  --type Bar
```

## 📊 Supported Chart Types

- **Bar Charts**: Vertical and horizontal bars with categorical data
- **Line Charts**: Time series and trend data with multiple series
- **Pie Charts**: Proportional data with percentage calculations

## 🔑 Key Features

- ✅ Automatic chart type detection
- ✅ Multi-orientation OCR (0°, ±45°, ±90°)
- ✅ Tilted/rotated X-axis label support
- ✅ Multi-series line chart support
- ✅ Legend matching with color detection
- ✅ Y-axis value calibration
- ✅ Web interface for interactive testing

## 📖 Recommended Reading Order

1. Start with [Architecture Overview](architecture.md) to understand the system
2. Follow [Installation Guide](installation.md) to set up your environment
3. Try [Quick Start](quick-start.md) to run your first extraction
4. Deep dive into [Pipeline Explanation](pipeline.md) to understand the flow
5. Explore [Model Details](models.md) for technical understanding

## 🤝 Contributing

See individual documentation files for detailed explanations of each component.

## 📄 License

See LICENSE file in the root directory.
