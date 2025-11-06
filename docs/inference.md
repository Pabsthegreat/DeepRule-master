# Inference Guide - Using Trained Models

## Quick Start

### Single Image Inference

```python
from test_pipe_type_cloud import predict_chart

# Predict single chart
result = predict_chart(
    image_path="sample_images/chart.png",
    chart_type="auto"  # or "Bar", "Line", "Pie"
)

# Result structure
{
    "chart_type": "Bar",
    "data": [
        {"category": "2020", "label": "Revenue", "value": 150000},
        {"category": "2021", "label": "Revenue", "value": 180000},
        ...
    ],
    "csv_path": "output/chart_result.csv"
}
```

### Batch Processing

```python
import glob
from test_pipe_type_cloud import predict_chart

# Process all images in folder
image_files = glob.glob("input_folder/*.png")

for img_path in image_files:
    try:
        result = predict_chart(img_path, chart_type="auto")
        print(f"✅ {img_path}: {result['chart_type']}")
    except Exception as e:
        print(f"❌ {img_path}: {e}")
```

### Web Interface

```bash
# Start Django server
python manage.py runserver 0.0.0.0:8000

# Open browser
http://localhost:8000
```

Upload chart image and get structured data output.

## Installation

See [Installation Guide](installation.md) for setup instructions.

### Quick Check

```bash
# Verify installation
python -c "
from test_pipe_type_cloud import Pre_load_nets
Pre_load_nets('Bar')
print('✅ Models loaded successfully')
"
```

## API Reference

### Main Functions

#### `predict_chart()`

Extract structured data from chart image.

```python
def predict_chart(
    image_path: str,
    chart_type: str = "auto",
    output_dir: str = "output",
    debug: bool = False
) -> dict:
    """
    Args:
        image_path: Path to chart image
        chart_type: "auto", "Bar", "Line", or "Pie"
        output_dir: Directory for output CSV
        debug: Save debug images
        
    Returns:
        {
            "chart_type": str,
            "data": list[dict],
            "csv_path": str,
            "debug_dir": str (if debug=True)
        }
    """
```

**Example**:

```python
result = predict_chart(
    "charts/sales.png",
    chart_type="auto",
    output_dir="results",
    debug=True
)

print(f"Type: {result['chart_type']}")
print(f"Data points: {len(result['data'])}")
print(f"Output: {result['csv_path']}")
print(f"Debug images: {result['debug_dir']}")
```

#### `auto_detect_chart_type()`

Automatically detect chart type.

```python
def auto_detect_chart_type(image_path: str) -> str:
    """
    Args:
        image_path: Path to chart image
        
    Returns:
        "Bar", "Line", or "Pie"
    """
```

**Example**:

```python
chart_type = auto_detect_chart_type("unknown_chart.png")
print(f"Detected: {chart_type}")
```

#### `Pre_load_nets()`

Pre-load models into memory (cache).

```python
def Pre_load_nets(chart_type: str):
    """
    Args:
        chart_type: "Bar", "Line", or "Pie"
        
    Side effects:
        - Loads models into global cache
        - Moves models to GPU (if available)
    """
```

**Example**:

```python
# Pre-load for faster inference
Pre_load_nets("Bar")
Pre_load_nets("Line")

# Now predictions are faster
for img in images:
    predict_chart(img, chart_type="Bar")  # Uses cached model
```

## Chart-Specific Usage

### Bar Charts

```python
result = predict_chart("bar_chart.png", chart_type="Bar")

# Output CSV format:
# category,label,color,value
# 2020,Revenue,#FF0000,150000
# 2021,Revenue,#FF0000,180000
# 2020,Profit,#0000FF,50000
# 2021,Profit,#0000FF,60000
```

**Requirements**:
- X-axis labels (categories)
- Y-axis scale (values)
- Bar detection

**Optional**:
- Legend (for multi-series)
- Title

### Line Charts

```python
result = predict_chart("line_chart.png", chart_type="Line")

# Output CSV format:
# category,label,color,x_pixel,y_pixel,value
# Jan,Sales,#0000FF,100,200,45.5
# Feb,Sales,#0000FF,150,180,52.3
# Mar,Sales,#0000FF,200,160,61.2
```

**Requirements**:
- Data points detection
- X-axis labels
- Y-axis scale

**Optional**:
- Multiple series
- Legend

### Pie Charts

```python
result = predict_chart("pie_chart.png", chart_type="Pie")

# Output CSV format:
# label,color,percentage
# Product A,#FF0000,35.5
# Product B,#00FF00,28.3
# Product C,#0000FF,36.2
```

**Requirements**:
- Sector detection
- Legend (labels)

**Optional**:
- Percentage labels on sectors

## Advanced Options

### Custom Configuration

```python
import json

# Load default config
with open("config/CornerNetCls.json") as f:
    config = json.load(f)

# Modify parameters
config["inference"]["top_k"] = 200  # Increase detections
config["inference"]["nms_threshold"] = 0.3  # Stricter NMS

# Save custom config
with open("config/custom.json", "w") as f:
    json.dump(config, f)

# Use custom config
# (Requires modifying code to load custom config)
```

### Debug Mode

Enable debug output to see intermediate results:

```python
result = predict_chart(
    "chart.png",
    debug=True
)

# Check debug directory
# debug_output/
#   xaxis_01_strip.png
#   xaxis_02_binary.png
#   xaxis_03_all_bboxes.png
#   xaxis_04_merged_bboxes.png
#   xaxis_05_bbox00_original.png
#   xaxis_05_bbox00_rot0.png
#   ...
```

**Debug images**:
1. X-axis extraction steps
2. Component detection bboxes
3. Data element detection
4. OCR results overlaid

### GPU vs CPU

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

# Force CPU (for testing)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
```

**Performance**:
- GPU: 0.5-1s per image
- CPU: 2-5s per image

### Batch Prediction

Optimize for multiple images:

```python
from test_pipe_type_cloud import Pre_load_nets

# 1. Pre-load models once
Pre_load_nets("Bar")

# 2. Process batch
results = []
for img_path in image_paths:
    result = predict_chart(img_path, chart_type="Bar")
    results.append(result)

# 3. Aggregate results
import pandas as pd
all_data = []
for r in results:
    df = pd.read_csv(r["csv_path"])
    df["source"] = r["image_path"]
    all_data.append(df)

combined = pd.concat(all_data, ignore_index=True)
combined.to_csv("all_results.csv", index=False)
```

## Output Formats

### CSV

Default output format.

```csv
category,label,color,value
2020,Revenue,#FF0000,150000
2021,Revenue,#FF0000,180000
```

### JSON

Convert CSV to JSON:

```python
import pandas as pd
import json

df = pd.read_csv("output.csv")
data = df.to_dict(orient="records")

with open("output.json", "w") as f:
    json.dump(data, f, indent=2)
```

Output:
```json
[
  {
    "category": "2020",
    "label": "Revenue",
    "color": "#FF0000",
    "value": 150000
  },
  {
    "category": "2021",
    "label": "Revenue",
    "color": "#FF0000",
    "value": 180000
  }
]
```

### Excel

Convert CSV to Excel:

```python
import pandas as pd

df = pd.read_csv("output.csv")
df.to_excel("output.xlsx", index=False)
```

## Error Handling

### Common Errors

#### 1. Model Not Found

```
FileNotFoundError: cache/nnet/CornerNetCls/CornerNetCls_50000.pkl
```

**Solution**: Download model checkpoints (see [Installation](installation.md))

#### 2. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**: Use CPU or smaller batch size

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
```

#### 3. No Plot Detected

```
ValueError: Plot region not detected
```

**Solution**: Check if image contains valid chart

```python
from test_pipe_type_cloud import testing
import cv2

image = cv2.imread("chart.png")
cls_info = testing(image, "CornerNetCls")

if 5 not in cls_info:
    print("❌ No plot area detected")
    print("Detected regions:", cls_info.keys())
```

#### 4. OCR Failure

```
Warning: No X-axis labels detected
```

**Solution**: Enable debug mode to see X-axis extraction

```python
result = predict_chart("chart.png", debug=True)
# Check debug_output/xaxis_*.png images
```

### Graceful Error Handling

```python
def safe_predict(image_path, chart_type="auto"):
    """
    Predict with error handling
    """
    try:
        result = predict_chart(image_path, chart_type=chart_type)
        return {"status": "success", "result": result}
        
    except FileNotFoundError as e:
        return {"status": "error", "message": f"File not found: {e}"}
        
    except ValueError as e:
        return {"status": "error", "message": f"Invalid chart: {e}"}
        
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {e}"}

# Usage
result = safe_predict("chart.png")
if result["status"] == "success":
    print(f"✅ Success: {result['result']['csv_path']}")
else:
    print(f"❌ Error: {result['message']}")
```

## Performance Tuning

### Optimize Inference Speed

```python
# 1. Pre-load models
Pre_load_nets("Bar")

# 2. Use GPU
import torch
assert torch.cuda.is_available(), "GPU not available"

# 3. Disable debug mode
result = predict_chart("chart.png", debug=False)

# 4. Batch process
for img in images:
    predict_chart(img, chart_type="Bar")  # Reuse loaded model
```

**Benchmark**:
| Configuration | Time per Image |
|---------------|----------------|
| CPU, cold start | ~5s |
| CPU, warm start | ~3s |
| GPU, cold start | ~2s |
| GPU, warm start | ~0.8s |

### Memory Optimization

```python
import gc
import torch

# Clear GPU memory
torch.cuda.empty_cache()
gc.collect()

# Process in smaller batches
batch_size = 10
for i in range(0, len(images), batch_size):
    batch = images[i:i+batch_size]
    for img in batch:
        predict_chart(img)
    
    # Clear memory after each batch
    torch.cuda.empty_cache()
```

## Integration Examples

### REST API

```python
from flask import Flask, request, jsonify
from test_pipe_type_cloud import predict_chart
import os

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    # Get uploaded file
    file = request.files["image"]
    
    # Save temporarily
    temp_path = f"temp/{file.filename}"
    file.save(temp_path)
    
    # Predict
    result = predict_chart(temp_path, chart_type="auto")
    
    # Clean up
    os.remove(temp_path)
    
    # Return JSON
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
```

**Usage**:
```bash
curl -X POST -F "image=@chart.png" http://localhost:5000/predict
```

### Command-Line Tool

```python
# cli.py
import argparse
from test_pipe_type_cloud import predict_chart

def main():
    parser = argparse.ArgumentParser(description="Chart data extraction")
    parser.add_argument("image", help="Path to chart image")
    parser.add_argument("--type", default="auto", choices=["auto", "Bar", "Line", "Pie"])
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    result = predict_chart(
        args.image,
        chart_type=args.type,
        output_dir=args.output,
        debug=args.debug
    )
    
    print(f"Chart type: {result['chart_type']}")
    print(f"Output CSV: {result['csv_path']}")

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
python cli.py chart.png --type auto --output results --debug
```

### Python Package

```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="deeprule",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.7.0",
        "opencv-python>=4.5.0",
        "pytesseract>=0.3.0",
        "numpy>=1.19.0",
        "pandas>=1.2.0",
    ],
    entry_points={
        "console_scripts": [
            "deeprule=cli:main",
        ],
    },
)
```

**Install**:
```bash
pip install -e .
```

**Use**:
```bash
deeprule chart.png --type Bar
```

## Testing

### Unit Tests

```python
# test_inference.py
import unittest
from test_pipe_type_cloud import predict_chart, auto_detect_chart_type

class TestInference(unittest.TestCase):
    
    def test_bar_chart(self):
        result = predict_chart("test/bar.png", chart_type="Bar")
        self.assertEqual(result["chart_type"], "Bar")
        self.assertGreater(len(result["data"]), 0)
    
    def test_auto_detect(self):
        chart_type = auto_detect_chart_type("test/line.png")
        self.assertEqual(chart_type, "Line")
    
    def test_invalid_image(self):
        with self.assertRaises(FileNotFoundError):
            predict_chart("nonexistent.png")

if __name__ == "__main__":
    unittest.main()
```

**Run tests**:
```bash
python -m unittest test_inference.py
```

### Integration Tests

```python
# test_integration.py
import os
import glob
from test_pipe_type_cloud import predict_chart

def test_all_samples():
    """
    Test all sample images
    """
    sample_dir = "sample_images"
    images = glob.glob(f"{sample_dir}/*.png")
    
    results = []
    for img_path in images:
        try:
            result = predict_chart(img_path, chart_type="auto")
            results.append({
                "image": os.path.basename(img_path),
                "status": "✅ Success",
                "type": result["chart_type"]
            })
        except Exception as e:
            results.append({
                "image": os.path.basename(img_path),
                "status": f"❌ Error: {e}",
                "type": None
            })
    
    # Print summary
    print("\n=== Test Summary ===")
    for r in results:
        print(f"{r['status']}: {r['image']} ({r['type']})")
    
    success_count = sum(1 for r in results if "✅" in r["status"])
    print(f"\nSuccess rate: {success_count}/{len(results)}")

if __name__ == "__main__":
    test_all_samples()
```

## Troubleshooting

See [Troubleshooting Guide](troubleshooting.md) for common issues and solutions.

### Quick Checks

```python
# 1. Check models loaded
from test_pipe_type_cloud import Pre_load_nets
Pre_load_nets("Bar")
print("✅ Models loaded")

# 2. Check GPU
import torch
print(f"GPU available: {torch.cuda.is_available()}")

# 3. Check Tesseract
import pytesseract
print(f"Tesseract version: {pytesseract.get_tesseract_version()}")

# 4. Check dependencies
import cv2, numpy, pandas
print("✅ All dependencies installed")
```

## Next Steps

- [Training Guide](training.md) - Train custom models
- [Pipeline Details](pipeline.md) - Understand the flow
- [X-axis Extraction](xaxis-extraction.md) - Deep dive into OCR
- [Configuration](configuration.md) - Customize parameters
