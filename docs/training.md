# Training Guide - Custom Model Training

## Overview

This guide explains how to train custom models for chart component detection and data extraction.

## Training Architecture

### Two-Stage Training

```
Stage 1: Component Detection (CornerNetCls)
┌────────────────────────────────────────┐
│ Input: Chart images                    │
│ Output: Bboxes for:                    │
│   - Title region                       │
│   - X-axis region                      │
│   - Y-axis region                      │
│   - Plot area                          │
│   - Legend region                      │
└────────────────────────────────────────┘

Stage 2: Chart-Specific Detection
┌────────────────────────────────────────┐
│ Bar Charts (CornerNetPureBar)          │
│   Input: Chart images                  │
│   Output: Individual bar bboxes        │
├────────────────────────────────────────┤
│ Line Charts (CornerNetLine)            │
│   Input: Chart images                  │
│   Output: Data point coordinates       │
├────────────────────────────────────────┤
│ Pie Charts (CornerNetPurePie)          │
│   Input: Chart images                  │
│   Output: Sector centers + edges       │
└────────────────────────────────────────┘
```

## Dataset Preparation

### Data Requirements

**Minimum Dataset Size**:
- Component detection: 500+ annotated charts
- Chart-specific: 200+ per chart type

**Recommended Dataset Size**:
- Component detection: 2000+ charts
- Chart-specific: 1000+ per type

**Data Distribution**:
```
Training set: 80%
Validation set: 10%
Test set: 10%
```

### Annotation Format

See [Dataset Documentation](dataset.md) for detailed format specification.

#### COCO JSON Structure

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "chart_001.png",
      "height": 800,
      "width": 1200
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 5,
      "bbox": [100, 50, 1000, 600],
      "area": 600000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "title"},
    {"id": 2, "name": "x_axis"},
    {"id": 3, "name": "y_axis"},
    {"id": 5, "name": "plot_area"},
    {"id": 6, "name": "legend"}
  ]
}
```

### Annotation Tools

#### LabelMe (Recommended)

```bash
# Install
pip install labelme

# Launch
labelme images/ \
  --output annotations/ \
  --labels title,x_axis,y_axis,plot_area,legend
```

**Workflow**:
1. Open image
2. Draw rectangle around each component
3. Assign label (title, x_axis, etc.)
4. Save JSON
5. Convert to COCO format

#### VGG Image Annotator (VIA)

Web-based tool: https://www.robots.ox.ac.uk/~vgg/software/via/

**Export**: JSON format
**Convert**: Use `scripts/via_to_coco.py`

### Data Augmentation

Applied during training:

```python
# In db/datasets.py

augmentations = [
    # 1. Random crop (preserve labels)
    RandomCrop(
        min_scale=0.8,
        max_scale=1.0
    ),
    
    # 2. Random horizontal flip
    RandomHorizontalFlip(prob=0.5),
    
    # 3. Color jitter
    ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    
    # 4. Random rotation
    RandomRotation(degrees=5),
    
    # 5. Gaussian noise
    GaussianNoise(mean=0, std=0.01),
]
```

**Augmentation multiplier**: Effective dataset size = real size × 5

## Configuration

### Config File Structure

Example: `config/CornerNetCls.json`

```json
{
  "system": {
    "batch_size": 8,
    "chunk_sizes": [8],
    "snapshot_name": "CornerNetCls",
    "data_dir": "./",
    "cache_dir": "./cache",
    "result_dir": "./results",
    "tar_data_dir": "Cls"
  },
  "db": {
    "train_split": "train",
    "val_split": "val",
    "test_split": "test"
  },
  "training": {
    "learning_rate": 0.00025,
    "decay_rate": 10,
    "max_iter": 50000,
    "val_iter": 1000,
    "snapshot_step": 5000
  },
  "nnet": {
    "input_size": [511, 511],
    "output_sizes": [[128, 128]],
    "corner_pooling": true,
    "backbone": "hourglass",
    "num_stacks": 2
  }
}
```

### Key Parameters

#### Training Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 8 | Images per batch |
| `learning_rate` | 0.00025 | Initial learning rate |
| `decay_rate` | 10 | LR decay factor |
| `max_iter` | 50000 | Total iterations |
| `val_iter` | 1000 | Validation frequency |
| `snapshot_step` | 5000 | Checkpoint frequency |

#### Model Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `input_size` | [511, 511] | Input image size |
| `output_sizes` | [[128, 128]] | Output heatmap size |
| `backbone` | hourglass | Feature extractor |
| `num_stacks` | 2 | Hourglass stacks |
| `corner_pooling` | true | Enable corner pooling |

#### Data Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `train_split` | train | Training set name |
| `val_split` | val | Validation set name |
| `data_dir` | ./ | Dataset root |
| `tar_data_dir` | Cls | Annotations folder |

## Training Process

### Step 1: Prepare Dataset

```bash
# 1. Organize data
mkdir -p Cls/images
mkdir -p Cls/annotations

# 2. Split dataset
python scripts/split_dataset.py \
  --input annotations/ \
  --output Cls/annotations/ \
  --train 0.8 \
  --val 0.1 \
  --test 0.1

# Generates:
# Cls/annotations/instances_train.json
# Cls/annotations/instances_val.json
# Cls/annotations/instances_test.json
```

### Step 2: Configure Training

```bash
# Copy base config
cp config/CornerNetCls.json config/MyModel.json

# Edit config
vim config/MyModel.json
```

**Essential edits**:
```json
{
  "system": {
    "snapshot_name": "MyModel",  # Change name
    "tar_data_dir": "Cls"         # Point to data
  }
}
```

### Step 3: Compile Extensions

```bash
# Navigate to external directory
cd external

# Compile corner pooling layers
python setup.py build_ext --inplace

# Compile NMS
python setup.py install

cd ..
```

**Verify compilation**:
```python
from external.corner_pool import TopPool, BottomPool
from external.nms import soft_nms
print("✅ Extensions compiled")
```

### Step 4: Start Training

```bash
# Train CornerNetCls (component detection)
python train_chart.py CornerNetCls

# Train chart-specific model
python train_chart.py CornerNetPureBar  # Bar charts
python train_chart.py CornerNetLine     # Line charts
python train_chart.py CornerNetPurePie  # Pie charts
```

**Training output**:
```
Iteration 100/50000, Loss: 2.5432
Iteration 200/50000, Loss: 2.3145
...
Iteration 1000/50000, Loss: 1.8234
Running validation...
Val Loss: 1.9123, mAP: 0.6543
Saving checkpoint: cache/nnet/MyModel/MyModel_1000.pkl
```

### Step 5: Monitor Training

#### TensorBoard (Optional)

```bash
# Log training metrics
tensorboard --logdir=./logs

# Open browser
http://localhost:6006
```

#### Manual Monitoring

```python
# Check training log
tail -f logs/training.log

# Plot loss curve
python scripts/plot_loss.py --log logs/training.log
```

### Step 6: Evaluate Model

```bash
# Test on validation set
python test_pipeline.py \
  --config config/MyModel.json \
  --checkpoint cache/nnet/MyModel/MyModel_50000.pkl \
  --split val

# Output:
# mAP: 0.8542
# Precision: 0.8734
# Recall: 0.8321
```

## Training from Scratch

### Component Detection (CornerNetCls)

```bash
# 1. Prepare annotations
python scripts/prepare_cls_data.py \
  --images data/charts/ \
  --output Cls/

# 2. Train
python train_chart.py CornerNetCls

# 3. Evaluate
python test_pipeline.py CornerNetCls
```

**Expected training time**: 24-48 hours on GPU

### Bar Chart Detection

```bash
# 1. Prepare bar annotations
python scripts/prepare_bar_data.py \
  --images data/bar_charts/ \
  --output Bar/

# 2. Train
python train_chart.py CornerNetPureBar

# 3. Evaluate
python test_pipeline.py CornerNetPureBar
```

**Expected training time**: 12-24 hours on GPU

### Line Chart Detection

Similar to bar charts, but uses line annotations:

```bash
python scripts/prepare_line_data.py --images data/line_charts/ --output line/
python train_chart.py CornerNetLine
python test_pipeline.py CornerNetLine
```

### Pie Chart Detection

```bash
python scripts/prepare_pie_data.py --images data/pie_charts/ --output pie/
python train_chart.py CornerNetPurePie
python test_pipeline.py CornerNetPurePie
```

## Fine-Tuning Pre-trained Models

### Load Pre-trained Weights

```python
# In train_chart.py

def load_pretrained(nnet, checkpoint_path):
    """
    Load pre-trained weights
    """
    checkpoint = torch.load(checkpoint_path)
    
    # Load state dict
    nnet.load_state_dict(checkpoint['state_dict'])
    
    # Optionally reset optimizer
    optimizer = torch.optim.Adam(
        nnet.parameters(),
        lr=0.00025
    )
    
    return nnet, optimizer

# Usage
nnet = load_model()
nnet, optimizer = load_pretrained(
    nnet,
    "cache/nnet/CornerNetCls/CornerNetCls_50000.pkl"
)

# Continue training
train(nnet, optimizer, start_iter=50000)
```

### Transfer Learning

```python
# Freeze early layers
for name, param in nnet.named_parameters():
    if "backbone" in name:
        param.requires_grad = False

# Only train head layers
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, nnet.parameters()),
    lr=0.0001  # Lower LR for fine-tuning
)
```

**Use cases**:
- Limited training data (< 500 images)
- Domain adaptation (e.g., medical charts)
- Faster convergence

## Loss Functions

### CornerNet Loss

```python
# Focal loss for keypoint detection
def focal_loss(pred, gt, alpha=2, beta=4):
    """
    Args:
        pred: Predicted heatmap [B, C, H, W]
        gt: Ground truth heatmap [B, C, H, W]
        
    Returns:
        Focal loss value
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()
    
    neg_weights = torch.pow(1 - gt, beta)
    
    loss = 0
    
    # Positive loss
    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    
    # Negative loss
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    
    return loss

# Total loss
total_loss = focal_loss(tl_heat, tl_gt) + focal_loss(br_heat, br_gt) + \
             pull_loss + push_loss + off_loss
```

### Loss Components

| Component | Purpose | Weight |
|-----------|---------|--------|
| Top-left focal | Detect top-left corners | 1.0 |
| Bottom-right focal | Detect bottom-right corners | 1.0 |
| Pull loss | Group corner pairs | 0.1 |
| Push loss | Separate different objects | 0.1 |
| Offset loss | Refine corner positions | 1.0 |

## Data Sampling

### Positive/Negative Ratio

```python
# In db/datasets.py

def sample_data(annotations):
    """
    Balance positive and negative samples
    """
    pos_samples = [a for a in annotations if a['category_id'] > 0]
    neg_samples = [a for a in annotations if a['category_id'] == 0]
    
    # Keep 1:3 ratio
    num_pos = len(pos_samples)
    num_neg = min(num_pos * 3, len(neg_samples))
    
    neg_samples = random.sample(neg_samples, num_neg)
    
    return pos_samples + neg_samples
```

### Hard Negative Mining

```python
def hard_negative_mining(predictions, gt, ratio=0.3):
    """
    Select hard negative samples
    """
    # Find false positives
    false_pos = (predictions > 0.5) & (gt == 0)
    
    # Sort by confidence
    scores = predictions[false_pos]
    sorted_idx = scores.argsort(descending=True)
    
    # Keep top ratio
    num_hard = int(len(sorted_idx) * ratio)
    hard_idx = sorted_idx[:num_hard]
    
    return hard_idx
```

## Training Tips

### Learning Rate Schedule

```python
def adjust_learning_rate(optimizer, iteration, initial_lr=0.00025):
    """
    Step decay learning rate
    """
    if iteration < 5000:
        lr = initial_lr
    elif iteration < 20000:
        lr = initial_lr / 2
    elif iteration < 35000:
        lr = initial_lr / 10
    else:
        lr = initial_lr / 100
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr
```

### Gradient Clipping

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(nnet.parameters(), max_norm=10.0)
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for iteration in range(max_iter):
    with autocast():
        loss = compute_loss(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Benefits**:
- 2x faster training
- 50% less GPU memory
- Same accuracy

### Checkpoint Management

```python
def save_checkpoint(nnet, optimizer, iteration, path):
    """
    Save training checkpoint
    """
    torch.save({
        'iteration': iteration,
        'state_dict': nnet.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, path)

def load_checkpoint(nnet, optimizer, path):
    """
    Load training checkpoint
    """
    checkpoint = torch.load(path)
    nnet.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['iteration']
```

## Evaluation Metrics

### Mean Average Precision (mAP)

```python
def compute_mAP(predictions, ground_truth, iou_threshold=0.5):
    """
    Compute mAP at IoU threshold
    """
    # Sort by confidence
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    
    tp = []
    fp = []
    
    for pred in predictions:
        # Find matching ground truth
        match = find_best_match(pred, ground_truth, iou_threshold)
        
        if match:
            tp.append(1)
            fp.append(0)
            ground_truth.remove(match)  # Already matched
        else:
            tp.append(0)
            fp.append(1)
    
    # Compute precision-recall curve
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    recalls = tp_cumsum / len(ground_truth)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    # Compute average precision
    ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
    
    return ap
```

### Per-Category Metrics

```python
# Component detection
mAP = {
    "title": 0.92,
    "x_axis": 0.88,
    "y_axis": 0.86,
    "plot_area": 0.95,
    "legend": 0.84,
    "overall": 0.89
}

# Bar detection
mAP = {
    "bar": 0.91,
    "overall": 0.91
}
```

## Troubleshooting Training Issues

### Issue 1: Loss Not Decreasing

**Symptoms**:
- Loss stays constant
- No improvement after 1000+ iterations

**Solutions**:
1. Check learning rate (too low?)
2. Verify data loading (correct labels?)
3. Reduce batch size (gradient accumulation issue?)
4. Check loss function (implemented correctly?)

### Issue 2: Overfitting

**Symptoms**:
- Train loss decreases
- Val loss increases

**Solutions**:
1. Add data augmentation
2. Reduce model capacity
3. Add dropout layers
4. Early stopping

```python
# Early stopping
best_val_loss = float('inf')
patience = 0
max_patience = 5

for epoch in range(num_epochs):
    train_loss = train_one_epoch()
    val_loss = validate()
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience = 0
        save_checkpoint()
    else:
        patience += 1
        if patience >= max_patience:
            print("Early stopping triggered")
            break
```

### Issue 3: Out of Memory

**Symptoms**:
- CUDA out of memory error

**Solutions**:
1. Reduce batch size
2. Reduce input size
3. Use gradient checkpointing
4. Use mixed precision

```python
# Reduce batch size
config["system"]["batch_size"] = 4  # Instead of 8

# Reduce input size
config["nnet"]["input_size"] = [384, 384]  # Instead of [511, 511]
```

### Issue 4: Slow Training

**Symptoms**:
- < 1 iteration per second

**Solutions**:
1. Use GPU (check CUDA availability)
2. Increase num_workers for data loading
3. Use mixed precision
4. Profile bottlenecks

```python
import torch.profiler

with torch.profiler.profile() as prof:
    for i in range(100):
        output = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

## Advanced Training

### Multi-GPU Training

```python
# DataParallel (simple)
if torch.cuda.device_count() > 1:
    nnet = torch.nn.DataParallel(nnet)

# DistributedDataParallel (faster)
import torch.distributed as dist

dist.init_process_group(backend='nccl')
nnet = torch.nn.parallel.DistributedDataParallel(nnet)
```

### Custom Loss Function

```python
class CustomLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, pred, gt, **kwargs):
        # Your custom loss logic
        loss = ...
        return loss

# Use custom loss
criterion = CustomLoss()
loss = criterion(outputs, targets)
```

### Learning Rate Finder

```python
def find_lr(nnet, train_loader, optimizer, start_lr=1e-7, end_lr=10):
    """
    Find optimal learning rate
    """
    lrs = []
    losses = []
    
    lr = start_lr
    while lr < end_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        loss = train_one_batch(nnet, train_loader, optimizer)
        
        lrs.append(lr)
        losses.append(loss.item())
        
        lr *= 1.1  # Increase by 10%
    
    # Plot loss vs LR
    import matplotlib.pyplot as plt
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.show()

# Usage
find_lr(nnet, train_loader, optimizer)
# Pick LR at steepest descent
```

## Next Steps

- [Dataset Documentation](dataset.md) - Annotation format details
- [Model Architecture](models.md) - CornerNet details
- [Inference Guide](inference.md) - Use trained models
- [Configuration](configuration.md) - Parameter tuning
