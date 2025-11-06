# DeepRule API

## Quick Start

### Web UI
```
http://localhost:8000/
```

### API Endpoint
```
POST http://localhost:8000/api/extract
```

### API Documentation
```
http://localhost:8000/api/docs
```

## Features

✅ **Professional UI** - Modern, responsive design with drag-and-drop
✅ **API Endpoint** - RESTful API for programmatic access
✅ **Debug Mode** - Optional debug flag for X-axis extraction troubleshooting
✅ **Memory-efficient** - Processes images in memory (no intermediate files unless debug=true)
✅ **Multi-chart Support** - Bar, Line, and Pie charts
✅ **Auto-detection** - Automatically detects chart type
✅ **CORS Enabled** - Cross-origin requests supported

## Recent Improvements

### UI Enhancements
- Modern gradient design with smooth animations
- Drag-and-drop file upload
- Real-time file name display
- Loading spinner during processing
- Responsive layout for mobile devices
- Professional color scheme and typography

### Performance Optimizations
- X-axis extraction now works in memory by default
- Debug images only saved when `debug=true`
- Reduced file I/O operations
- Faster processing times

### API Features
- RESTful endpoint at `/api/extract`
- CORS support for cross-origin requests
- Comprehensive error handling
- Detailed response with metadata
- Support for debug mode via parameter

## Usage Examples

### cURL
```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@chart.png" \
  -F "chart_type=Auto"
```

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/api/extract",
    files={"file": open("chart.png", "rb")},
    data={"chart_type": "Auto"}
)
print(response.json())
```

### JavaScript
```javascript
const formData = new FormData();
formData.append('file', file);
formData.append('chart_type', 'Auto');

fetch('http://localhost:8000/api/extract', {
    method: 'POST',
    body: formData
})
.then(r => r.json())
.then(data => console.log(data));
```

## Debug Mode

Enable debug mode to save intermediate X-axis extraction images:

```bash
curl -X POST http://localhost:8000/api/extract \
  -F "file=@chart.png" \
  -F "chart_type=Auto" \
  -F "debug=true"
```

Debug images saved to `debug_output/`:
- `xaxis_01_strip.png` - Extracted X-axis region
- `xaxis_02_binary.png` - Binary threshold
- `xaxis_02b_dilated.png` - Morphological dilation
- `xaxis_03_all_bboxes.png` - All detected bounding boxes
- `xaxis_04_merged_bboxes.png` - Merged bounding boxes
- `xaxis_05_bbox##_*.png` - Individual crops with rotations (0°, ±45°, ±90°)

## Response Format

```json
{
  "status": "success",
  "chart_type": "Bar",
  "data": [
    {
      "category": "2020",
      "label": "Revenue",
      "value": 150000,
      "color": "#FF0000"
    }
  ],
  "metadata": {
    "titles": {
      "chart": "Annual Revenue",
      "value_axis": "Amount ($)",
      "category_axis": "Year"
    },
    "y_axis_range": {
      "min": 0,
      "max": 200000,
      "source": "detected"
    }
  }
}
```

## Running the Server

```bash
# Activate virtual environment
source .venv/bin/activate

# Run Django server
python manage.py runserver 0.0.0.0:8000
```

## Configuration

Edit `settings.py` for:
- Debug mode
- Allowed hosts
- Static file paths
- CORS settings

## Troubleshooting

### X-axis labels not detected
- Enable debug mode: `debug=true`
- Check `debug_output/` images
- Verify labels are in X-axis region
- Try manual chart type selection

### Low confidence OCR
- Ensure image resolution is sufficient (min 800x600)
- Check for image artifacts or noise
- Try different Tesseract languages

### API CORS errors
- CORS is enabled by default
- Check browser console for details
- Verify request headers

## Documentation

- **Full Documentation**: See `/docs` folder
- **API Docs**: http://localhost:8000/api/docs
- **Installation**: `docs/installation.md`
- **X-axis Extraction**: `docs/xaxis-extraction.md`
- **Pipeline**: `docs/pipeline.md`
