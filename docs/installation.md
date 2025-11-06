# Installation Guide

## Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 18.04+) or macOS
- **GPU**: NVIDIA GPU with CUDA support (required)
- **CUDA**: Version 10.0 or higher
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: 5GB for models and cache

### Software Requirements

- Python 3.7-3.13
- CUDA Toolkit
- cuDNN
- GCC/G++ compiler (for building extensions)

## Installation Methods

### Method 1: Conda Environment (Recommended - Legacy)

**Note**: The original conda environment file (`DeepRule.txt`) may be outdated.

```bash
# Create environment from package list
conda create --name DeepRule --file DeepRule.txt

# Activate environment
conda activate DeepRule
```

### Method 2: Pip Installation (Current - 2023)

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements-2023.txt
```

## Step-by-Step Setup

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/DeepRule-master.git
cd DeepRule-master
```

### 2. Install Python Dependencies

```bash
# Using pip (recommended)
pip install -r requirements-2023.txt

# Key packages:
# - torch==1.13.0 (or compatible)
# - torchvision
# - opencv-python
# - pytesseract
# - django==5.2.7
# - Pillow
# - numpy
# - scipy
# - pycocotools
```

### 3. Install Tesseract OCR

#### Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
sudo apt-get install libtesseract-dev
```

#### macOS (Homebrew):
```bash
brew install tesseract
```

#### Verify Installation:
```bash
tesseract --version
# Should show: tesseract 4.x.x or 5.x.x
```

### 4. Compile Corner Pooling Layers

Corner pooling is a critical component for the CornerNet architecture.

```bash
cd models/py_utils/_cpools/
python setup.py build_ext --inplace
cd ../../..
```

**Expected Output**:
```
running build_ext
building 'top_pool' extension
...
copying build/lib.*/top_pool.*.so -> .
```

**Troubleshooting**:
- If you get compilation errors, ensure GCC is installed
- Check that CUDA paths are correctly set
- Verify Python development headers are installed

### 5. Compile NMS (Non-Maximum Suppression)

```bash
cd external/
make
cd ..
```

**Expected Output**:
```
python setup.py build_ext --inplace
running build_ext
building 'nms' extension
...
```

**If Make Fails**:
```bash
# Manually compile
cd external/
python setup.py build_ext --inplace
cd ..
```

### 6. Install MS COCO API

```bash
pip install pycocotools
```

**Alternative (if above fails)**:
```bash
pip install git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI
```

### 7. Download Trained Models

Download pre-trained model weights from Hugging Face:

**Link**: [asbljy/DeepRuleDataset](https://huggingface.co/datasets/asbljy/DeepRuleDataset/tree/main)

```bash
# Download and extract to project root
# The archive should contain:
# - cache/nnet/CornerNetCls/CornerNetCls_50000.pkl
# - cache/nnet/CornerNetPureBar/CornerNetPureBar_50000.pkl
# - cache/nnet/CornerNetLine/CornerNetLine_50000.pkl
# - cache/nnet/CornerNetPurePie/CornerNetPurePie_50000.pkl
```

**Verify Structure**:
```bash
ls -la cache/nnet/
# Should show 4 directories with .pkl files
```

### 8. (Optional) Download Training Data

Only needed if you plan to train models:

```bash
# Download from Hugging Face
# Extract to project root
# Should create: Bar/, Cls/, line/, pie/ folders
```

## Verification

### Test Installation

#### 1. Test Imports:
```bash
python -c "import torch; print(torch.cuda.is_available())"
# Should print: True
```

#### 2. Test Tesseract:
```bash
python -c "import pytesseract; print(pytesseract.get_tesseract_version())"
# Should print version number
```

#### 3. Test Corner Pooling:
```bash
python -c "from models.py_utils._cpools import TopPool; print('Corner pooling OK')"
# Should print: Corner pooling OK
```

#### 4. Test NMS:
```bash
python -c "from external import nms; print('NMS OK')"
# Should print: NMS OK
```

### Run Quick Test

```bash
# Start Django server
python manage.py runserver 0.0.0.0:8000

# Open browser: http://localhost:8000
# Upload a sample chart image
# Verify output appears
```

## Common Issues and Solutions

### Issue 1: CUDA Not Available

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA
python -c "import torch; print(torch.version.cuda)"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue 2: Corner Pooling Compilation Fails

**Error**: `error: command 'gcc' failed`

**Solutions**:
```bash
# Ubuntu
sudo apt-get install build-essential python3-dev

# macOS
xcode-select --install

# Verify GCC
gcc --version
```

### Issue 3: Tesseract Not Found

**Error**: `TesseractNotFoundError`

**Solutions**:
```bash
# Set TESSDATA_PREFIX
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Or in Python:
import os
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'
```

### Issue 4: Missing Model Files

**Error**: `FileNotFoundError: cache/nnet/CornerNetCls/CornerNetCls_50000.pkl`

**Solution**: Download models from Hugging Face and extract correctly

### Issue 5: Import Errors

**Error**: `ModuleNotFoundError: No module named 'pycocotools'`

**Solution**: Install missing package
```bash
pip install pycocotools
```

## Environment Variables

Create a `.env` file or set these in your shell:

```bash
# CUDA settings
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Tesseract
export TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

# Django
export DJANGO_SETTINGS_MODULE=server_match.settings
export DEBUG=True
```

## Docker Installation (Alternative)

If you prefer containerization:

```bash
# TODO: Create Dockerfile
# This would encapsulate all dependencies
```

## Post-Installation Checklist

- [ ] Python environment activated
- [ ] All pip packages installed
- [ ] Tesseract OCR installed and working
- [ ] Corner pooling compiled successfully
- [ ] NMS compiled successfully
- [ ] CUDA available and working
- [ ] Model weights downloaded and in `cache/nnet/`
- [ ] Django server starts without errors
- [ ] Can process a sample chart successfully

## Next Steps

- [Quick Start Guide](quick-start.md) - Run your first extraction
- [API Reference](api-reference.md) - Using the web interface
- [Troubleshooting](troubleshooting.md) - More detailed problem solving
