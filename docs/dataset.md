# Dataset Format and Structure

## Overview

DeepRule uses the **COCO (Common Objects in Context)** annotation format for training data. This format is widely used in object detection tasks and provides a standardized way to store bounding box annotations.

## Directory Structure

```
DeepRule-master/
├── Bar/
│   └── annotations/
│       └── instancesBar(1031)_val2019.json
├── Cls/
│   └── annotations/
│       └── instancesCls(*)_val2019.json
├── line/
│   └── annotations/
│       └── instancesLine(*)_val2019.json
└── pie/
    └── annotations/
        └── instancesPie(1008)_val2019.json
```

**Note**: These folders are only needed for **training**. For inference, you only need the trained model weights in `cache/nnet/`.

## Annotation Format

### General COCO Structure

```json
{
  "images": [
    {
      "id": 12345,
      "file_name": "chart_001.png",
      "width": 800,
      "height": 600
    }
  ],
  "annotations": [
    {
      "id": 67890,
      "image_id": 12345,
      "category_id": 0,
      "bbox": [...],
      "area": 1500.0
    }
  ],
  "categories": [
    {
      "id": 0,
      "name": "bar"
    }
  ]
}
```

## Chart-Specific Formats

### 1. Bar Chart Annotations

**File**: `Bar/annotations/instancesBar(1031)_val2019.json`

**Bbox Format**: `[x, y, width, height]`

```json
{
  "image_id": 12345,
  "category_id": 0,
  "bbox": [100.0, 150.0, 50.0, 200.0],
  "area": 10000.0,
  "id": 433872
}
```

**Interpretation**:
- `x, y`: Top-left corner of the bar
- `width`: Bar width
- `height`: Bar height
- `area`: Width × height

**Example**:
```
A bar starting at (100, 150), 
50 pixels wide, 200 pixels tall
     ↓
  ┌─────┐ ← (100, 150)
  │     │
  │ Bar │ 50px wide
  │     │
  │     │
  └─────┘
    ↑
  200px tall
```

### 2. Pie Chart Annotations

**File**: `pie/annotations/instancesPie(1008)_val2019.json`

**Bbox Format**: `[center_x, center_y, edge1_x, edge1_y, edge2_x, edge2_y]`

```json
{
  "image_id": 74999,
  "category_id": 0,
  "bbox": [135.0, 60.0, 132.0, 60.0, 134.0, 130.0],
  "area": 105.02630551355209,
  "id": 433872
}
```

**Interpretation**:
- `center_x, center_y`: Center of the pie chart
- `edge1_x, edge1_y`: First edge point of the sector
- `edge2_x, edge2_y`: Second edge point of the sector

**Example**:
```
       edge1 (132, 60)
          ╱
         ╱
        ╱
   center (135, 60)
        ╲
         ╲
          ╲
       edge2 (134, 130)
```

The three points define a pie sector (slice).

### 3. Line Chart Annotations

**File**: `line/annotations/instancesLine(*)_val2019.json`

**Bbox Format**: `[x1, y1, x2, y2, ..., xn, yn]` (variable length)

```json
{
  "image_id": 120596,
  "category_id": 0,
  "bbox": [137.0, 131.0, 174.0, 113.0, 210.0, 80.0, 247.0, 85.0],
  "area": 0,
  "id": 288282
}
```

**Interpretation**:
- Sequence of (x, y) data points for a line series
- Variable length array (2n elements for n points)

**Example**:
```
Points: (137, 131) → (174, 113) → (210, 80) → (247, 85)

     ●  Point 1: (137, 131)
      ╲
       ●  Point 2: (174, 113)
        ╲
         ●  Point 3: (210, 80)
        ╱
       ●  Point 4: (247, 85)
```

**Special Files**:
- `instancesLineClsEx`: Used for training LineCls (line classification for series separation)

### 4. Classification Annotations (Cls)

**File**: `Cls/annotations/instancesCls(*)_val2019.json`

**Bbox Format**: `[x, y, width, height]` (standard bounding box)

```json
{
  "image_id": 12345,
  "category_id": 5,
  "bbox": [50.0, 100.0, 500.0, 400.0],
  "area": 200000.0,
  "id": 111
}
```

**Category IDs**:
```python
{
  1: "Title",
  2: "X-axis label area",
  3: "Y-axis label area", 
  5: "Plot area (data region)",
  6: "Legend"
}
```

**Example**:
```
┌───────────────────────────────────┐
│          Chart Title (cat=1)      │
├───────────────────────────────────┤
│ Y │                               │
│   │      Plot Area (cat=5)        │
│ A │                               │
│ x │                               │
│ i │          Legend (cat=6)       │
│ s │          ┌─────────┐          │
│   │          │ Series A│          │
│ 3 │          │ Series B│          │
│   │          └─────────┘          │
├───┴───────────────────────────────┤
│        X-axis labels (cat=2)      │
└───────────────────────────────────┘
```

## Dataset Statistics

### Training Datasets (as mentioned in README)

- **Bar Charts**: 1,031 annotated images
- **Pie Charts**: 1,008 annotated images
- **Line Charts**: Variable (check your downloaded dataset)
- **Cls (Components)**: All chart types

### Data Source

**Hugging Face**: [asbljy/DeepRuleDataset](https://huggingface.co/datasets/asbljy/DeepRuleDataset/tree/main)

## Image Requirements

### Supported Formats
- PNG (recommended)
- JPG/JPEG
- BMP

### Recommended Specifications
- **Resolution**: 400x300 to 1920x1080 pixels
- **Aspect Ratio**: 4:3 or 16:9 (standard chart proportions)
- **Color Mode**: RGB (color charts)
- **File Size**: < 5MB per image

### Quality Guidelines
- Clear, readable text
- Sufficient contrast
- No excessive noise or artifacts
- Complete chart visible (not cropped)

## Creating Custom Annotations

### Tools
- [COCO Annotator](https://github.com/jsbroks/coco-annotator) (web-based)
- [LabelMe](https://github.com/wkentaro/labelme) (convert to COCO format)
- [VGG Image Annotator (VIA)](https://www.robots.ox.ac.uk/~vgg/software/via/)

### Annotation Guidelines

#### For Bar Charts:
1. Draw tight bounding box around each bar
2. Include the full bar height (from baseline to top)
3. Separate stacked bars if applicable
4. Use category_id = 0 for all bars

#### For Pie Charts:
1. Click center of pie
2. Click first edge of sector (clockwise)
3. Click second edge of sector
4. Ensure points form a valid sector
5. One annotation per sector

#### For Line Charts:
1. Click each data point along the line
2. Follow the line from left to right
3. One annotation per line series
4. Include all visible data points

#### For Classification:
1. Draw bounding box for each component
2. Assign correct category_id:
   - 1: Title
   - 2: X-axis
   - 3: Y-axis
   - 5: Plot area
   - 6: Legend

## Data Augmentation (During Training)

The training pipeline applies these augmentations:

```python
Augmentations:
- Random horizontal flip
- Random vertical flip  
- Random rotation (±15°)
- Random scale (0.8-1.2x)
- Random crop
- Color jittering
- Gaussian noise
```

**Note**: Augmentation only happens during training, not inference.

## Validation Split

Datasets are split into:
- **Training**: `train2019`
- **Validation**: `val2019`
- **Testing**: `test2019`

The code automatically loads the appropriate split based on configuration.

## Cached Data

**Location**: `cache/` directory

```
cache/
├── cls_val2019.pkl      # Cached Cls validation data
├── bar_val2019.pkl      # Cached Bar validation data
├── line_val2019.pkl     # Cached Line validation data
└── pie_val2019.pkl      # Cached Pie validation data
```

**Purpose**: 
- Faster data loading on subsequent runs
- Preprocessed annotations
- Reduces I/O overhead

**When to Clear**:
- After changing annotation files
- After updating dataset split
- If cache is corrupted

```bash
# Clear cache
rm -rf cache/*.pkl
```

## Data Preprocessing Pipeline

```
Raw Annotations (JSON)
        ↓
Parse COCO format
        ↓
Filter by split (train/val/test)
        ↓
Convert bbox format
        ↓
Create image-annotation pairs
        ↓
Cache to .pkl files
        ↓
Load batches during training
```

## Common Issues

### Issue 1: Missing Annotation Files
**Error**: `FileNotFoundError: instancesBar_val2019.json`

**Solution**: Download dataset from Hugging Face and extract to correct location

### Issue 2: Corrupted Cache
**Error**: `EOFError` or `pickle.UnpicklingError`

**Solution**: Delete .pkl files and let them regenerate

### Issue 3: Wrong Bbox Format
**Error**: `ValueError: invalid literal for int()`

**Solution**: Verify bbox format matches chart type specification

## Next Steps

- [Training Guide](training.md) - How to train with your dataset
- [Model Details](models.md) - Understanding model architecture
- [Pipeline Explanation](pipeline.md) - How data flows through the system
