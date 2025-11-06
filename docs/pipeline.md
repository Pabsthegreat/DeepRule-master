# Pipeline Explanation - End-to-End Flow

## Overview

This document explains the complete data flow from input image to structured CSV output.

## High-Level Pipeline

```
┌──────────────┐
│ Input Image  │
│ chart.png    │
└──────┬───────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 1: Chart Type Detection          │
│  - Auto: Hough Circles + Lines         │
│  - Manual: User selection              │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 2: Model Loading & Caching       │
│  - CornerNetCls (always)               │
│  - Chart-specific model (Bar/Line/Pie) │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 3: Component Detection           │
│  - Plot area bounding box              │
│  - Title region                        │
│  - X-axis region                       │
│  - Y-axis region                       │
│  - Legend region                       │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 4: Data Element Detection        │
│  Bar: Individual bar bounding boxes    │
│  Line: Data point coordinates          │
│  Pie: Sector center + edges            │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 5: Full Image OCR                │
│  - Tesseract on entire image           │
│  - Extract all text + bounding boxes   │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 6: X-axis Label Extraction       │
│  - Detect text regions below plot      │
│  - Multi-orientation OCR (0°, ±45°, ±90°)│
│  - Confidence-based selection          │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 7: Y-axis Calibration            │
│  - Detect Y-axis tick numbers          │
│  - Build pixel→value mapping           │
│  - Linear regression for interpolation │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 8: Legend Matching               │
│  - Extract legend colors               │
│  - Match to data series colors         │
│  - Assign labels to series             │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 9: Rule-Based Processing         │
│  - Apply chart-specific rules          │
│  - Map labels to data elements         │
│  - Validate and structure data         │
└──────┬─────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────┐
│  Step 10: CSV Generation               │
│  - Create DataFrame                    │
│  - Format columns                      │
│  - Export to CSV                       │
└──────┬─────────────────────────────────┘
       │
       ▼
┌──────────────┐
│ Output CSV   │
│ data.csv     │
└──────────────┘
```

## Detailed Step-by-Step

### Step 1: Chart Type Detection

**Function**: `auto_detect_chart_type()`

**Logic**:
```python
1. Convert to grayscale
2. Run Hough Circle detection
   - If circles found → Pie Chart
3. Run Canny edge detection
4. Run Hough Line detection
5. Count horizontal lines (≥ W/6 width)
   - If ≥ 3 lines → Line Chart
6. Default → Bar Chart
```

**Parameters**:
```python
HoughCircles:
  - minRadius: H/8
  - maxRadius: H/2
  - minDist: H/4

HoughLinesP:
  - threshold: 100
  - minLineLength: W/4
  - maxLineGap: 10
```

**Output**: Chart type string ("Bar", "Line", or "Pie")

### Step 2: Model Loading

**Function**: `Pre_load_nets()`

**Process**:
```python
1. Check if models already cached in memory
2. If not cached:
   a. Load CornerNetCls (always needed)
   b. Load chart-specific model:
      - Bar: CornerNetPureBar
      - Line: CornerNetLine  
      - Pie: CornerNetPurePie
3. Move models to GPU (if available)
4. Set models to eval mode
5. Cache in global dictionary
```

**Files Loaded**:
```
config/CornerNetCls.json → Model configuration
cache/nnet/CornerNetCls/CornerNetCls_50000.pkl → Model weights
```

**Performance**: 
- First load: 2-5 seconds
- Subsequent loads: < 0.1 seconds (cached)

### Step 3: Component Detection

**Function**: `testing()` from `testfile/test_CornerNetCls.py`

**Process**:
```python
1. Preprocess image:
   - Resize to model input size
   - Normalize pixel values
   - Convert to tensor
   
2. Forward pass through CornerNetCls:
   - Detect top-left corners
   - Detect bottom-right corners
   - Pair corners to form bboxes
   
3. Apply NMS (Non-Maximum Suppression):
   - Remove overlapping detections
   - Keep highest confidence boxes
   
4. Group detections by category:
   - Category 1: Title
   - Category 2: X-axis  
   - Category 3: Y-axis
   - Category 5: Plot area (MOST IMPORTANT)
   - Category 6: Legend
```

**Output**: `cls_info` dictionary
```python
{
  1: [x1, y1, x2, y2],  # Title bbox
  2: [x1, y1, x2, y2],  # X-axis bbox
  3: [x1, y1, x2, y2],  # Y-axis bbox
  5: [x1, y1, x2, y2],  # Plot bbox
  6: [x1, y1, x2, y2],  # Legend bbox
}
```

### Step 4: Data Element Detection

#### For Bar Charts

**Function**: `testing()` from `testfile/test_CornerNetPureBar.py`

```python
1. Forward pass through CornerNetPureBar
2. Detect bar bounding boxes
3. Apply NMS to remove duplicates
4. Return list of bars: [(x1, y1, x2, y2), ...]
```

**Output**: List of bar rectangles

#### For Line Charts

**Function**: `testing()` from `testfile/test_CornerNetLine.py`

```python
1. Forward pass through CornerNetLine
2. Detect line data points
3. Group points by series (category_id)
4. Return points with series assignment
```

**Output**: 
```python
[
  {"bbox": [x1, y1, w1, h1], "category_id": 0},  # Series 0, Point 1
  {"bbox": [x2, y2, w2, h2], "category_id": 0},  # Series 0, Point 2
  {"bbox": [x3, y3, w3, h3], "category_id": 1},  # Series 1, Point 1
  ...
]
```

#### For Pie Charts

**Function**: `testing()` from `testfile/test_CornerNetPurePie.py`

```python
1. Forward pass through CornerNetPurePie
2. Detect pie sectors (center + 2 edges)
3. Calculate angles and percentages
4. Return sector information
```

**Output**: List of sectors with angles

### Step 5: Full Image OCR

**Function**: `ocr_result_full_image()`

```python
1. Load image
2. Convert BGR → RGB
3. Run Tesseract OCR:
   pytesseract.image_to_data(
       image,
       lang="eng",
       output_type=pytesseract.Output.DICT
   )
4. Extract for each detected word:
   - text: "Label"
   - bbox: [x, y, x+w, y+h]
5. Filter empty strings
```

**Output**: List of word dictionaries
```python
[
  {"text": "2020", "bbox": [100, 50, 150, 70]},
  {"text": "Sales", "bbox": [200, 30, 280, 50]},
  ...
]
```

### Step 6: X-axis Label Extraction ⭐

**This is the most complex step!**

**Function**: `ocr_xaxis_region_with_rotation()`

#### Sub-step 6.1: Detect Text Regions

**Function**: `detect_xaxis_text_regions()`

```python
1. Extract region below plot:
   y_start = plot_y2 + 5  # Skip 5px to avoid plot border
   y_end = image_height
   x_start = plot_x1 - 5
   x_end = plot_x2 + 20

2. Binary thresholding (Otsu method):
   - Convert to grayscale
   - Adaptive threshold
   
3. Morphological dilation (connect characters):
   - Horizontal kernel: 3x1
   - Vertical kernel: 1x8
   - Ellipse kernel: 3x3
   
4. Find contours:
   - Detect connected regions
   - Filter noise (< 5x5 pixels)
   
5. Merge nearby bboxes:
   - Calculate overlap percentage
   - Merge if overlap ≥ 70% AND vertical_gap < 10px
   - Keep separate otherwise
```

**Output**: List of text region bboxes

#### Sub-step 6.2: Multi-Orientation OCR

For each detected text region:

```python
1. Extract crop with padding:
   - pad_x = 25px
   - pad_y_top = 15px
   - pad_y_bottom = 0px

2. Determine if text is horizontal:
   is_horizontal = crop_width > crop_height * 1.2

3. If horizontal:
   - Try normal orientation first
   - If confidence < 50, try rotations
   
4. If tilted/vertical:
   - Try ALL 5 orientations:
     * normal (0°)
     * 45° CCW
     * 90° CCW
     * 45° CW
     * 90° CW

5. Run Tesseract on each orientation:
   - Extract text + confidence scores
   - Track best result
   
6. Select highest confidence result
```

**Output**: List of X-axis labels with positions
```python
[
  {"text": "Jan-2020", "bbox": [100, 450, 150, 480]},
  {"text": "Feb-2020", "bbox": [180, 450, 230, 480]},
  ...
]
```

### Step 7: Y-axis Calibration

**Function**: `fit_y_to_value_function()`

```python
1. Find Y-axis numbers:
   - Look for text on left side (x < 60 pixels)
   - Must be 1-2 digit numbers
   - Extract value and y-position

2. Identify anchor points:
   - Top tick (min Y pixel → max value)
   - Bottom tick (max Y pixel → min value)

3. Build linear regression:
   y_value = a * y_pixel + b
   
4. Return calibration function
```

**Output**: Function that maps pixel Y → data value

### Step 8: Legend Matching

**Function**: `find_legend_pairs()` + `closest_legend()`

```python
1. Find legend items:
   - Extract text from legend region
   - Sample color at each text position
   
2. For each data series:
   - Sample color from data element
   - Find closest legend color (Euclidean distance in RGB)
   - Assign legend label to series
```

**Output**: Series labels
```python
{
  0: "Revenue",
  1: "Profit",
  2: "Cost"
}
```

### Step 9: Rule-Based Processing

#### Bar Charts (`RuleGroup/Bar.py`)

```python
1. Map labels to bars:
   - Match X-axis labels to bar centers
   - Assign closest label to each bar
   
2. Calculate bar values:
   - Use Y-calibration: y_to_val(bar_top_y)
   - Handle baseline offset
   
3. Match colors to legend:
   - Sample bar color
   - Find matching legend entry
```

#### Line Charts (`RuleGroup/Line.py`)

```python
1. Group points by series:
   - Use category_id from detection
   
2. Sort points by X position (left to right)

3. Map X-labels to points:
   - Find closest label for each point
   
4. Convert Y-pixels to values:
   - Apply y_to_val() function
   
5. Match series colors to legend
```

#### Pie Charts (`RuleGroup/Pie.py`)

```python
1. Calculate sector angles:
   - Use center + edge points
   - Compute angle span
   
2. Calculate percentages:
   - angle / 360 * 100
   
3. Match legend labels to sectors:
   - Color-based matching
```

### Step 10: CSV Generation

```python
1. Create DataFrame:
   rows = []
   for each data element:
       row = {
           "category": x_label,
           "label": series_label,
           "color": hex_color,
           "value": data_value
       }
       rows.append(row)
   
2. Save to CSV:
   df = pd.DataFrame(rows)
   df.to_csv("output.csv", index=False)
```

**Output Example**:

Bar Chart CSV:
```csv
category,label,color,value
2020,Revenue,#FF0000,150000
2021,Revenue,#FF0000,180000
2022,Revenue,#FF0000,200000
```

Line Chart CSV:
```csv
category,label,color,x_pixel,y_pixel,value
Jan,Sales,#0000FF,100,200,45.5
Feb,Sales,#0000FF,150,180,52.3
```

## Performance Metrics

| Step | Time (avg) | Notes |
|------|------------|-------|
| Type Detection | 0.1s | Hough transforms |
| Model Loading | 2-5s (first time) | Cached after |
| Component Detection | 0.5s | GPU inference |
| Data Detection | 0.5-1s | Depends on complexity |
| Full OCR | 0.5-2s | Depends on text density |
| X-axis OCR | 1-3s | Multi-orientation |
| Y-axis Calibration | 0.1s | Simple regression |
| Legend Matching | 0.1s | Color comparison |
| Rule Processing | 0.1s | Python logic |
| CSV Export | 0.05s | Pandas |
| **Total** | **3-10s** | Varies by chart |

## Error Handling

Each step includes error handling:

```python
try:
    result = process_step()
except Exception as e:
    print(f"Step failed: {e}")
    return default_value
```

Common failure points:
- No plot region detected → Cannot proceed
- No Y-axis ticks found → Cannot calibrate
- OCR fails → Return empty labels
- Model loading fails → Check file paths

## Debug Mode

Enable debug output:

```python
detect_xaxis_text_regions(..., debug=True)
```

Generates debug images in `debug_output/`:
- `xaxis_01_strip.png` - Extracted region
- `xaxis_02_binary.png` - Thresholded
- `xaxis_02b_dilated.png` - Dilated
- `xaxis_03_all_bboxes.png` - Detected bboxes
- `xaxis_04_merged_bboxes.png` - After merging
- `xaxis_05_bbox##_*.png` - Individual crops + rotations

## Next Steps

- [X-axis Extraction Details](xaxis-extraction.md) - Deep dive into Step 6
- [Model Details](models.md) - Neural network architectures
- [API Reference](api-reference.md) - Using the system programmatically
