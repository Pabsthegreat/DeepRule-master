# Architecture Overview

## System Design

DeepRule is built on a modular architecture that separates concerns into distinct components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Web Interface                            │
│                    (Django Server - port 8000)                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Core Processing Pipeline                      │
│                  (test_pipe_type_cloud.py)                      │
├─────────────────────────────────────────────────────────────────┤
│  1. Chart Type Detection (Auto or Manual)                       │
│  2. Model Loading & Caching                                     │
│  3. Neural Network Inference                                    │
│  4. OCR Text Extraction                                         │
│  5. Rule-Based Post-Processing                                  │
│  6. Data Structuring & Export                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
            ┌────────────────┼────────────────┐
            │                │                │
            ▼                ▼                ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │ Bar Detection│  │Line Detection│  │ Pie Detection│
    │   Models     │  │   Models     │  │   Models     │
    └──────────────┘  └──────────────┘  └──────────────┘
```

## Core Components

### 1. Neural Network Models

**Location**: `models/` directory

DeepRule uses CornerNet-based architectures for object detection:

- **CornerNetCls** - Classifies and locates chart components
  - Detects: plot area, title, X-axis, Y-axis, legend
  
- **CornerNetPureBar** - Detects individual bars
  - Output: Bounding boxes of bars
  
- **CornerNetLine** - Detects line data points
  - Output: Sequence of (x, y) coordinates per line series
  
- **CornerNetPurePie** - Detects pie sectors
  - Output: Center point + two edge points per sector

**Key Architecture Features**:
- Corner pooling layers for detecting bounding boxes
- Multi-scale feature extraction
- Hourglass network backbone
- Focal loss for handling class imbalance

### 2. Database Layer

**Location**: `db/` directory

Handles data loading and preprocessing:

- `coco.py` - COCO-format dataset loader
- `datasets.py` - Dataset factory and registry
- `detection.py` - Detection-specific data handling

**Responsibilities**:
- Load training annotations
- Batch preparation
- Data augmentation
- Caching for faster loading

### 3. OCR Engine

**Location**: `test_pipe_type_cloud.py` (functions: `ocr_result_full_image`, `ocr_xaxis_region_with_rotation`)

**Technology**: Pytesseract (Tesseract OCR)

**Capabilities**:
- Full image OCR for general text
- Multi-orientation OCR (0°, ±45°, ±90°)
- Confidence-based selection
- Bounding box detection
- Individual text region processing

**X-axis Label Extraction Pipeline**:
```
Plot Region Detection
        ↓
Extract X-axis Strip (below plot)
        ↓
Binary Thresholding (Otsu)
        ↓
Morphological Dilation (connect characters)
        ↓
Contour Detection (find text regions)
        ↓
Bbox Grouping (overlap-based merging)
        ↓
Individual Rotation & OCR (try 5 angles)
        ↓
Confidence-based Selection
```

### 4. Rule-Based Processing

**Location**: `RuleGroup/` directory

Post-processing rules for each chart type:

- **Bar.py** - Bar chart rules
  - Map labels to bars
  - Calculate bar heights
  - Handle stacked bars
  
- **Line.py** - Line chart rules
  - Series separation
  - Point-to-label mapping
  - Interpolation
  
- **Pie.py** - Pie chart rules
  - Angle calculation
  - Percentage computation
  - Sector labeling

### 5. Web Interface

**Location**: `server_match/` directory

Django-based web application:

- `views.py` - Request handlers
- `urls.py` - URL routing
- `templates/` - HTML templates
- `static/` - CSS, JavaScript, assets

**Features**:
- Image upload interface
- Chart type selection (Auto/Bar/Line/Pie)
- Real-time processing
- Result visualization
- CSV download

## Data Flow

### Training Flow

```
Annotated Images (COCO format)
        ↓
Dataset Loader (db/coco.py)
        ↓
Data Augmentation
        ↓
Batch Creation
        ↓
Neural Network (models/)
        ↓
Loss Calculation (Focal Loss)
        ↓
Backpropagation
        ↓
Model Checkpoints (cache/nnet/)
```

### Inference Flow

```
Input Image
        ↓
[Auto-Detection?] ──Yes──> Hough Circles + Lines
        │                         ↓
        No                  Chart Type Decision
        │                         │
        └─────────────────────────┘
                ↓
    Load Cached Models
                ↓
    Run Classification Model (CornerNetCls)
                ↓
    Detect: Plot, Title, Axes, Legend
                ↓
    Run Chart-Specific Model
    (Bar/Line/Pie)
                ↓
    Detect Data Elements
    (Bars/Points/Sectors)
                ↓
    Full Image OCR (Tesseract)
                ↓
    Extract Text Elements
                ↓
    X-axis Specialized OCR
    (Multi-orientation)
                ↓
    Y-axis Calibration
    (Tick detection + regression)
                ↓
    Legend Matching
    (Color-based pairing)
                ↓
    Rule-Based Processing
    (RuleGroup/)
                ↓
    Structure Data
                ↓
    Generate CSV Output
```

## Key Design Decisions

### 1. CornerNet-based Detection

**Why?** 
- Better for detecting objects with irregular shapes (bars, line segments)
- Anchor-free approach reduces hyperparameter tuning
- Corner pooling captures edge information effectively

### 2. Multi-orientation OCR

**Why?**
- Charts often have rotated axis labels (especially X-axis)
- Single-orientation OCR misses vertical/tilted text
- Confidence-based selection ensures best result

### 3. Rule-Based Post-Processing

**Why?**
- Neural networks detect elements but don't understand relationships
- Rules enforce constraints (e.g., bar-label alignment, series grouping)
- Domain knowledge improves accuracy

### 4. Caching Strategy

**Why?**
- Models are large (~200M parameters)
- Loading from disk is slow
- Cache models in memory after first load
- Pickle files (.pkl) for faster dataset loading

## Module Dependency Graph

```
test_pipe_type_cloud.py (Main Pipeline)
    │
    ├─> models/ (Neural Networks)
    │   ├─> py_utils/ (Corner Pooling, Loss)
    │   └─> nnet/ (Factory)
    │
    ├─> db/ (Dataset Handling)
    │   └─> datasets.py
    │
    ├─> RuleGroup/ (Post-processing)
    │   ├─> Bar.py
    │   ├─> Line.py
    │   └─> Pie.py
    │
    ├─> utils/ (Utilities)
    │   ├─> image.py
    │   └─> ...
    │
    ├─> external/ (NMS)
    │   └─> nms.pyx
    │
    └─> testfile/ (Testing Utils)
        └─> test_*.py
```

## Performance Characteristics

- **Model Loading**: 2-5 seconds (first time), instant (cached)
- **Inference**: 1-3 seconds per chart
- **OCR**: 0.5-2 seconds (depends on text density)
- **Total Pipeline**: 3-8 seconds per chart

## Scalability Considerations

- **GPU Required**: Models are GPU-optimized (CUDA)
- **Memory**: ~4GB GPU memory for inference
- **Concurrent Requests**: Limited by GPU memory and model caching
- **Batch Processing**: Supported via command-line interface

## Extension Points

1. **New Chart Types**: Add new model in `models/` + rules in `RuleGroup/`
2. **Better OCR**: Replace Tesseract with cloud APIs (Azure CV, Google Vision)
3. **Model Improvements**: Retrain with more data or better augmentation
4. **Auto-Detection**: Enhance with ML-based type classifier
5. **Export Formats**: Add JSON, XML, or database outputs

## Technology Stack

- **Deep Learning**: PyTorch 1.1+
- **Computer Vision**: OpenCV, PIL
- **OCR**: Tesseract (via pytesseract)
- **Web Framework**: Django 5.2
- **Data Format**: COCO JSON annotations
- **Deployment**: Python 3.13, CUDA support

## Next Steps

- [Model Details](models.md) - Deep dive into neural architectures
- [Pipeline Explanation](pipeline.md) - Step-by-step processing flow
- [Training Guide](training.md) - How to train models
