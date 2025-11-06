# X-axis Label Extraction - Multi-Orientation OCR

## Problem Statement

Chart X-axis labels can appear in various orientations:
- **Horizontal**: Standard left-to-right text
- **Tilted 45°**: Common for crowded labels
- **Vertical 90°**: Space-saving layout

**Challenge**: Standard OCR expects horizontal text and fails on rotated labels.

## Solution: Multi-Orientation OCR Pipeline

### Overview

```
┌─────────────────┐
│  Chart Image    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Step 1: Extract X-axis Region   │
│ - Below plot area               │
│ - Full width + padding          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Step 2: Detect Text Regions     │
│ - Binary threshold              │
│ - Morphological dilation        │
│ - Find contours                 │
│ - Merge nearby bboxes           │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Step 3: For Each Text Region    │
│ ┌─────────────────────────────┐ │
│ │ 3a. Determine Orientation   │ │
│ │     (Horizontal vs Tilted)  │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │ 3b. Try Multiple Rotations  │ │
│ │     (0°, ±45°, ±90°)        │ │
│ └─────────────────────────────┘ │
│ ┌─────────────────────────────┐ │
│ │ 3c. Select Best Result      │ │
│ │     (Highest confidence)    │ │
│ └─────────────────────────────┘ │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│ Step 4: Sort & Return Labels    │
│ - Sort left to right            │
│ - Remove duplicates             │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│ X-axis Labels   │
│ ["2020", "2021",│
│  "2022", ...]   │
└─────────────────┘
```

## Step 1: Extract X-axis Region

### Function: `ocr_xaxis_region_with_rotation()`

### Region Definition

```python
# Start 5 pixels below plot to avoid plot border
y_start = plot_y2 + 5  

# Extend to bottom of image
y_end = image_height

# Left: 5px before plot
x_start = max(0, plot_x1 - 5)

# Right: 20px after plot
x_end = min(image_width, plot_x2 + 20)

# Extract region
xaxis_strip = image[y_start:y_end, x_start:x_end]
```

### Why 5px Below?

**Problem**: Many charts have white gridlines at the plot border.

```
┌───────────────────────┐
│       Plot Area       │
│                       │
│                       │
└───────────────────────┘  ← White line interferes!
   ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲
   Labels here
```

**Solution**: Skip first 5 pixels
```
┌───────────────────────┐
│       Plot Area       │
│                       │
│                       │
└───────────────────────┘
     5px skip
   ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲ ▲  ← Start here
   Labels
```

## Step 2: Detect Text Regions

### Function: `detect_xaxis_text_regions()`

### Sub-step 2.1: Binary Thresholding

```python
# Convert to grayscale
gray = cv2.cvtColor(xaxis_strip, cv2.COLOR_BGR2GRAY)

# Otsu's method - automatic threshold selection
_, binary = cv2.threshold(
    gray,
    0, 255,
    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)
```

**Result**: White text on black background

### Sub-step 2.2: Morphological Dilation

**Purpose**: Connect characters into words/labels

```python
# Horizontal dilation (connect letters)
kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
dilated = cv2.dilate(binary, kernel_h, iterations=1)

# Vertical dilation (connect lines of multi-line labels)
kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))
dilated = cv2.dilate(dilated, kernel_v, iterations=1)

# Ellipse dilation (smooth connections)
kernel_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilated = cv2.dilate(dilated, kernel_e, iterations=1)
```

**Visual Effect**:

Before dilation:
```
J  a  n     2  0  2  0
```

After dilation:
```
█████████   █████████
 Jan-2020     block
```

### Dilation Kernel Tuning

**History of Changes**:

| Version | Kernels | Issue |
|---------|---------|-------|
| v1 | (5,1), (1,15), (7,7) | Over-merging tilted labels |
| v2 | (3,1), (1,10), (5,5) | Still some merging |
| **v3** | **(3,1), (1,8), (3,3)** | **✅ Correct** |

**Why Thinner Kernels?**

Tilted labels at 45° need special handling:

```
Horizontal labels:
┌────┐  ┌────┐  ┌────┐
│2020│  │2021│  │2022│
└────┘  └────┘  └────┘
(Safe to merge horizontally)

Tilted labels:
  2020
    \
     2021
       \
        2022
(Must NOT merge!)
```

Thick vertical dilation (15px) would merge tilted labels:
```
  2020
  ████  ← Dilates down
  ████     and merges!
   2021
   ████
    2022
```

Thin vertical dilation (8px) keeps them separate:
```
  2020
   ██   ← Connects within label
   
   2021  ← Stays separate
   ██
   
    2022
```

### Sub-step 2.3: Find Contours

```python
contours, _ = cv2.findContours(
    dilated,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

bboxes = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    
    # Filter noise
    if w < 5 or h < 5:
        continue
        
    bboxes.append([x, y, x+w, y+h])
```

### Sub-step 2.4: Merge Nearby Bboxes

**Purpose**: Combine labels split by dilation artifacts

**Algorithm**:

```python
def should_merge(box1, box2):
    """
    Merge if:
    1. Overlap ≥ 70%, AND
    2. Vertical gap < 10 pixels
    
    Both conditions MUST be true!
    """
    
    # Calculate overlap
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    intersection = x_overlap * y_overlap
    union = area1 + area2 - intersection
    
    iou = intersection / union
    
    # Vertical gap
    vertical_gap = abs(box1[1] - box2[1])
    
    # BOTH conditions required
    return (iou >= 0.7) and (vertical_gap < 10)

# Iterative merging
changed = True
while changed:
    changed = False
    for i in range(len(bboxes)):
        for j in range(i+1, len(bboxes)):
            if should_merge(bboxes[i], bboxes[j]):
                # Merge: take min/max coordinates
                merged = [
                    min(bboxes[i][0], bboxes[j][0]),
                    min(bboxes[i][1], bboxes[j][1]),
                    max(bboxes[i][2], bboxes[j][2]),
                    max(bboxes[i][3], bboxes[j][3])
                ]
                bboxes[i] = merged
                bboxes.pop(j)
                changed = True
                break
        if changed:
            break
```

**Why 70% Overlap + Vertical Gap?**

**Old approach** (50% overlap OR 5px gap):
```
  2020   ← 45° tilted
    \
     2021  ← Small gap merges them!
```

**New approach** (70% overlap AND < 10px gap):
```
  2020   ← Separate
    \      (no significant overlap)
     
     2021  ← Separate
```

## Step 3: Multi-Orientation OCR

### For Each Detected Text Region

#### Sub-step 3a: Determine Orientation

```python
# Extract crop with padding
pad_x = 25
pad_y_top = 15
pad_y_bottom = 0

x1 = max(0, bbox[0] - pad_x)
y1 = max(0, bbox[1] - pad_y_top)
x2 = min(strip_width, bbox[2] + pad_x)
y2 = min(strip_height, bbox[3] + pad_y_bottom)

crop = xaxis_strip[y1:y2, x1:x2]

# Check aspect ratio
width = crop.shape[1]
height = crop.shape[0]

is_horizontal = (width > height * 1.2)
```

**Classification**:
- **Horizontal**: width > 1.2 × height
  - Example: 120px wide × 30px tall = 4.0 ratio ✅
  
- **Tilted/Vertical**: width ≤ 1.2 × height
  - Example: 50px wide × 80px tall = 0.625 ratio ✅

#### Sub-step 3b: Try Multiple Rotations

##### Strategy 1: Horizontal Labels

```python
if is_horizontal:
    # 1. Try normal first (most likely)
    text, conf = ocr_with_tesseract(crop, angle=0)
    
    # 2. If low confidence, try rotations
    if conf < 50:
        for angle in [45, 90, -45, -90]:
            rotated = rotate_image(crop, angle)
            text_rot, conf_rot = ocr_with_tesseract(rotated)
            
            if conf_rot > conf:
                text = text_rot
                conf = conf_rot
```

**Rationale**: Horizontal labels are rarely rotated, so optimize for normal case.

##### Strategy 2: Tilted/Vertical Labels

```python
else:  # Tilted or vertical
    best_text = ""
    best_conf = 0
    
    # Try ALL orientations
    for angle in [0, 45, 90, -45, -90]:
        rotated = rotate_image(crop, angle)
        text, conf = ocr_with_tesseract(rotated)
        
        if conf > best_conf:
            best_text = text
            best_conf = conf
    
    text = best_text
    conf = best_conf
```

**Rationale**: Tilted labels could be at any angle, so try all possibilities.

#### Rotation Helper Function

```python
def rotate_image(image, angle):
    """
    Rotate image by arbitrary angle
    
    Args:
        image: Input image
        angle: Rotation angle (degrees, CCW positive)
        
    Returns:
        Rotated image with black padding
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    # Get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale=1.0)
    
    # Calculate new bounding box size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new size
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    # Rotate with black background
    rotated = cv2.warpAffine(
        image, M, (new_w, new_h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    
    return rotated
```

**Visual Example**:

Original (45° tilted):
```
    2
   0
  2
 0
```

Rotated -45° (corrected to horizontal):
```
2020
```

#### Sub-step 3c: OCR with Tesseract

```python
def ocr_with_tesseract(image):
    """
    Run Tesseract OCR and extract confidence
    """
    # Convert BGR → RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run Tesseract
    data = pytesseract.image_to_data(
        rgb,
        lang="eng",
        config="--psm 7",  # Single line mode
        output_type=pytesseract.Output.DICT
    )
    
    # Extract text and confidence
    texts = []
    confidences = []
    
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])
        
        if text and conf > 0:
            texts.append(text)
            confidences.append(conf)
    
    # Combine results
    full_text = " ".join(texts)
    avg_conf = np.mean(confidences) if confidences else 0
    
    return full_text, avg_conf
```

**Tesseract PSM Modes**:

| Mode | Description | Use Case |
|------|-------------|----------|
| 3 | Automatic (default) | General text |
| 6 | Uniform block | Paragraphs |
| **7** | **Single line** | **X-axis labels** ✅ |
| 8 | Single word | Individual words |
| 10 | Single character | OCR verification |

**Why PSM 7?** X-axis labels are typically single lines, so this mode optimizes accuracy.

### Sub-step 3d: Select Best Result

```python
results = []

for angle in angles_to_try:
    rotated = rotate_image(crop, angle)
    text, conf = ocr_with_tesseract(rotated)
    
    results.append({
        "angle": angle,
        "text": text,
        "confidence": conf
    })

# Sort by confidence (descending)
results.sort(key=lambda x: x["confidence"], reverse=True)

# Take best result
best_result = results[0]

print(f"Best: '{best_result['text']}' at {best_result['angle']}° "
      f"(confidence: {best_result['confidence']:.1f}%)")
```

**Example Output**:
```
Trying 0°: "|||" (conf: 12.3%)
Trying 45°: "20Z0" (conf: 45.7%)
Trying 90°: "2020" (conf: 92.1%) ← BEST
Trying -45°: "OZO" (conf: 23.4%)
Trying -90°: "0Z0Z" (conf: 34.2%)

Best: '2020' at 90° (confidence: 92.1%)
```

## Step 4: Post-Processing

### Sort Left to Right

```python
# Sort by x-coordinate
labels.sort(key=lambda x: x["bbox"][0])
```

### Remove Duplicates

```python
unique_labels = []
seen = set()

for label in labels:
    if label["text"] not in seen:
        unique_labels.append(label)
        seen.add(label["text"])
```

### Filter Noise

```python
# Remove very low confidence
labels = [l for l in labels if l["confidence"] > 20]

# Remove very short text
labels = [l for l in labels if len(l["text"]) >= 2]

# Remove non-alphanumeric
import re
labels = [l for l in labels if re.search(r'[a-zA-Z0-9]', l["text"])]
```

## Debug Mode

### Enable Debugging

```python
labels, bboxes = detect_xaxis_text_regions(
    xaxis_strip,
    debug=True,
    debug_dir="debug_output"
)
```

### Generated Debug Images

1. **`xaxis_01_strip.png`** - Raw X-axis region
2. **`xaxis_02_binary.png`** - After binary threshold
3. **`xaxis_02b_dilated.png`** - After dilation
4. **`xaxis_03_all_bboxes.png`** - All detected bboxes (red)
5. **`xaxis_04_merged_bboxes.png`** - After merging (green)
6. **`xaxis_05_bbox00_*.png`** - Individual crops:
   - `_original.png` - Extracted crop
   - `_rot0.png` - No rotation
   - `_rot45.png` - +45° rotation
   - `_rot90.png` - +90° rotation
   - `_rotn45.png` - -45° rotation
   - `_rotn90.png` - -90° rotation

### Debug Image Example

**`xaxis_04_merged_bboxes.png`**:
```
┌─────────────────────────────┐
│                             │
│  ┌────┐  ┌────┐  ┌────┐    │
│  │2020│  │2021│  │2022│    │
│  └────┘  └────┘  └────┘    │
│    ↑       ↑       ↑        │
│  Label1  Label2  Label3     │
│                             │
└─────────────────────────────┘
```

Each bbox drawn with:
- Green rectangle
- Label: "Label#"
- Confidence score

## Performance Optimization

### Caching

**Problem**: OCR is expensive (0.5-1s per label)

**Solution**: Cache results

```python
ocr_cache = {}

def ocr_with_cache(image_hash, image):
    if image_hash in ocr_cache:
        return ocr_cache[image_hash]
    
    result = ocr_with_tesseract(image)
    ocr_cache[image_hash] = result
    return result
```

### Parallel Processing

```python
from concurrent.futures import ThreadPoolExecutor

def ocr_all_labels(crops):
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(ocr_label, crops))
    return results
```

**Speedup**: 2-3x faster for charts with many labels

### Early Termination

```python
# If first attempt has high confidence, skip rotations
text, conf = ocr_with_tesseract(crop)

if conf > 90:
    return text, conf  # Good enough!

# Otherwise, try rotations...
```

**Speedup**: ~30% for mostly horizontal labels

## Common Issues & Solutions

### Issue 1: Labels Merging

**Symptom**: Multiple labels detected as one
```
"2020 2021 2022" instead of ["2020", "2021", "2022"]
```

**Solution**: Increase merge threshold from 50% → 70%

### Issue 2: Missing Labels

**Symptom**: Some labels not detected

**Causes**:
- Dilation too weak → increase kernel size
- Binary threshold wrong → try adaptive threshold
- Labels outside region → increase padding

**Debug**: Check `xaxis_03_all_bboxes.png` to see detected regions

### Issue 3: Wrong Orientation

**Symptom**: OCR returns gibberish
```
"OZOZ" instead of "2020"
```

**Solution**: Ensure all 5 rotations are tried, select highest confidence

**Debug**: Check individual rotation images (`_rot*.png`)

### Issue 4: Low Confidence

**Symptom**: All attempts have confidence < 50%

**Causes**:
- Image quality poor
- Text too small
- Background noise

**Solutions**:
- Upscale image 2x before OCR
- Apply Gaussian blur for denoising
- Use different Tesseract config

## Algorithm Parameters

### Tunable Parameters

```python
# Region extraction
PLOT_BOTTOM_SKIP = 5  # Skip 5px below plot

# Dilation kernels
KERNEL_H = (3, 1)  # Horizontal
KERNEL_V = (1, 8)  # Vertical
KERNEL_E = (3, 3)  # Ellipse

# Merging thresholds
OVERLAP_THRESHOLD = 0.7  # 70% IoU
VERTICAL_GAP_MAX = 10    # 10 pixels

# OCR padding
PAD_X = 25       # Horizontal padding
PAD_Y_TOP = 15   # Top padding
PAD_Y_BOTTOM = 0 # Bottom padding

# Orientation detection
HORIZONTAL_RATIO = 1.2  # width/height > 1.2 → horizontal

# Confidence thresholds
MIN_CONFIDENCE = 20     # Filter noise
SKIP_ROTATION_CONF = 90 # Skip rotations if > 90%

# Rotation angles
ANGLES = [0, 45, 90, -45, -90]  # Degrees
```

### Adjustment Guidelines

| Scenario | Parameter | Adjustment |
|----------|-----------|------------|
| Labels merging | `OVERLAP_THRESHOLD` | Increase (0.7 → 0.8) |
| Labels missing | Dilation kernels | Increase size |
| Wrong orientation | `ANGLES` | Add more angles |
| Low confidence | `PAD_X`, `PAD_Y_TOP` | Increase padding |
| Slow performance | `SKIP_ROTATION_CONF` | Lower (90 → 80) |

## Comparison with Alternatives

### Alternative 1: Rotate Entire Image

```python
# DON'T DO THIS
for angle in [0, 45, 90]:
    rotated_image = rotate(entire_image, angle)
    labels = ocr_full_image(rotated_image)
```

**Problems**:
- Rotates entire image (wasteful)
- May rotate plot area incorrectly
- 3x slower

### Alternative 2: Single Orientation

```python
# DON'T DO THIS
labels = ocr_region(xaxis_region)  # No rotation
```

**Problems**:
- Fails on tilted labels
- Only works for horizontal text

### Alternative 3: Deep Learning OCR

```python
# Possible but overkill
model = load_bert_ocr_model()
labels = model.predict(xaxis_region)
```

**Problems**:
- Much slower (10-20x)
- Requires training data
- Less accurate for diverse fonts

**Our approach** (multi-orientation Tesseract):
- Fast (1-3s total)
- No training required
- Works for all orientations ✅

## Next Steps

- [Pipeline Overview](pipeline.md) - Full system flow
- [Inference Guide](inference.md) - Using the system
- [Troubleshooting](troubleshooting.md) - Common issues
