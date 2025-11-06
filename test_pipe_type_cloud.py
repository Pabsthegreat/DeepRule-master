#!/usr/bin/env python
import os
import re
import io
import cv2
import json
import math
import base64
import torch
import numpy as np
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets
import importlib

from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBarRaw
from RuleGroup.LineQuiry import GroupQuiryRaw
from RuleGroup.LIneMatch import GroupLineRaw
from RuleGroup.Pie import GroupPie

import pytesseract
torch.backends.cudnn.benchmark = False

methods = None
_methods_cache = {}


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

def _encode_image_to_base64(pil_img):
    """Encode a PIL image as a PNG data URI."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d, exist_ok=True)

def _to_py(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    return o

def _to_box(coords):
    if coords is None:
        return None
    c = np.array(coords).reshape(-1)
    if c.size == 4:
        x1, y1, x2, y2 = map(float, c.tolist())
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    if c.size == 8:
        xs, ys = c[0::2], c[1::2]
        return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    if c.size >= 4:
        x1, y1, x2, y2 = map(float, c[:4].tolist())
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return None

def boxes_intersection_ratio(box_big, box_small):
    B, S = _to_box(box_big), _to_box(box_small)
    if B is None or S is None:
        return 0.0
    bx1, by1, bx2, by2 = B
    sx1, sy1, sx2, sy2 = S
    inter_x1 = max(bx1, sx1)
    inter_y1 = max(by1, sy1)
    inter_x2 = min(bx2, sx2)
    inter_y2 = min(by2, sy2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    small_area = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    return inter_area / small_area


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_net(testiter, cfg_name, data_dir, cache_dir, cuda_id=0):
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file) as f:
        configs = json.load(f)
    configs["system"].update({
        "snapshot_name": cfg_name,
        "data_dir": data_dir,
        "cache_dir": cache_dir,
        "result_dir": "result_dir",
        "tar_data_dir": "Cls",
    })
    system_configs.update_config(configs["system"])

    split = {"training": system_configs.train_split,
             "validation": system_configs.val_split,
             "testing": system_configs.test_split}["validation"]

    test_iter = system_configs.max_iter if testiter is None else testiter
    print(f"loading parameters at iteration: {test_iter}")
    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)

    nnet = NetworkFactory(db)
    print("loading parameters...")
    nnet.load_params(test_iter)
    if torch.cuda.is_available():
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet

def Pre_load_nets(chart_type, id_cuda, data_dir, cache_dir, force_reload=False):
    global methods
    cache_key = (chart_type, id_cuda, os.path.abspath(data_dir), os.path.abspath(cache_dir))
    if not force_reload and cache_key in _methods_cache:
        methods = _methods_cache[cache_key]
        return methods

    methods_local = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", data_dir, cache_dir, id_cuda)
    path_cls = "testfile.test_CornerNetCls"
    testing_cls = importlib.import_module(path_cls).testing
    methods_local["Cls"] = [db_cls, nnet_cls, testing_cls]

    if chart_type == "Bar":
        db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", data_dir, cache_dir, id_cuda)
        path_bar = "testfile.test_CornerNetPureBar"
        testing_bar = importlib.import_module(path_bar).testing
        methods_local["Bar"] = [db_bar, nnet_bar, testing_bar]
    
    elif chart_type == "Line":
        db_line, nnet_line = load_net(50000, "CornerNetLine", data_dir, cache_dir, id_cuda)
        path_line = "testfile.test_CornerNetLine"
        testing_line = importlib.import_module(path_line).testing
        methods_local["Line"] = [db_line, nnet_line, testing_line]
    
    elif chart_type == "Pie":
        db_pie, nnet_pie = load_net(50000, "CornerNetPurePie", data_dir, cache_dir, id_cuda)
        path_pie = "testfile.test_CornerNetPurePie"
        testing_pie = importlib.import_module(path_pie).testing
        methods_local["Pie"] = [db_pie, nnet_pie, testing_pie]

    _methods_cache[cache_key] = methods_local
    methods = methods_local
    return methods


# ------------------------------------------------------------------
# Auto-detection
# ------------------------------------------------------------------

def auto_detect_chart_type(image_path, data_dir, cache_dir, id_cuda=0):
    """
    Automatically detect chart type by running classification and checking 
    the detected elements using a lightweight approach.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return "Bar"  # Default fallback
        
        pil_img = Image.open(image_path).convert("RGB")
        
        # Load only the classifier model
        db_cls, nnet_cls = load_net(50000, "CornerNetCls", data_dir, cache_dir, id_cuda)
        path_cls = "testfile.test_CornerNetCls"
        testing_cls = importlib.import_module(path_cls).testing
        
        with torch.no_grad():
            cls_res = testing_cls(img, db_cls, nnet_cls, debug=False)
            tls, brs = cls_res[1], cls_res[2]
            _, raw_info = GroupCls(pil_img, tls, brs)
            cls_info = {int(k): _to_box(v) for k, v in raw_info.items()}
        
        # Analyze image characteristics
        H, W = img.shape[:2]
        
        # Check for circular shapes (pie chart indicator)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=H//4,
                                   param1=50, param2=30, minRadius=H//8, maxRadius=H//2)
        
        has_circles = circles is not None and len(circles[0]) > 0
        
        # Check for horizontal lines (line chart indicator)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=W//4, maxLineGap=10)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if line is mostly horizontal
                if abs(y2 - y1) < 20 and abs(x2 - x1) > W // 6:
                    horizontal_lines += 1
        
        # Decision logic
        if has_circles:
            return "Pie"
        elif horizontal_lines >= 3:  # Multiple horizontal lines suggest line chart
            return "Line"
        else:
            return "Bar"  # Default to bar chart
            
    except Exception as e:
        print(f"Auto-detection failed: {e}")
        return "Bar"  # Default fallback


# ------------------------------------------------------------------
# OCR helpers
# ------------------------------------------------------------------

def ocr_result_full_image(image_path):
    """
    OCR the full image for general text extraction.
    """
    os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    
    words = []
    
    # Normal orientation OCR
    data = pytesseract.image_to_data(pil, lang="eng", output_type=pytesseract.Output.DICT)
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({"text": txt, "bbox": [x, y, x + w, y + h]})
    
    return words

def extract_titles_from_clsinfo(cls_info, word_infos):
    out = {}
    for rid in [1, 2, 3]:
        if rid not in cls_info:
            continue
        rbox = _to_box(cls_info[rid])
        captured = [(w["text"], w["bbox"][1], w["bbox"][0]) for w in word_infos if boxes_intersection_ratio(rbox, w["bbox"]) > 0.5]
        captured.sort(key=lambda x: (x[1], x[2]))
        if captured:
            out[str(rid)] = " ".join([c[0] for c in captured])
    return out

def extract_axis_scale_from_clsinfo(image, cls_info):
    if 5 not in cls_info:
        return None, None
    plot = _to_box(cls_info[5])
    if plot is None:
        return None, None
    x1, y1, x2, y2 = map(int, plot)
    strip = image[y1:y2, max(0, x1 - 60):x1]
    if strip.size == 0:
        return None, None
    up = cv2.resize(strip, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    data = pytesseract.image_to_data(Image.fromarray(cv2.cvtColor(up, cv2.COLOR_BGR2RGB)),
                                     config="--psm 6", output_type=pytesseract.Output.DICT)
    nums = []
    for t in data["text"]:
        t = re.sub(r"[^0-9.\-]", "", t or "")
        if t and re.search(r"\d", t):
            try: nums.append(float(t))
            except: pass
    if not nums: return None, None
    return min(nums), max(nums)

def detect_xaxis_text_regions(image_path, plot_bbox, debug=False):
    """
    Detect individual text regions (bounding boxes) in the X-axis area using edge detection.
    Returns list of bounding boxes (x, y, w, h) in the X-axis strip.
    """
    img = cv2.imread(image_path)
    if img is None or plot_bbox is None:
        return [], None, (0, 0)
    
    x1_plot, y1_plot, x2_plot, y2_plot = plot_bbox
    H, W = img.shape[:2]
    
    # Extract region below plot
    # Add small buffer below plot line to avoid capturing plot border
    y_start = int(y2_plot) + 5  # Skip 5 pixels to avoid plot border/gridline
    y_end = H
    x_start = max(int(x1_plot)-5, 0)
    x_end = min(W, int(x2_plot) + 20)
    
    if y_start >= y_end or x_start >= x_end:
        return [], None, (0, 0)
    
    xaxis_strip = img[y_start:y_end, x_start:x_end].copy()
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STEP 1: X-AXIS REGION EXTRACTION")
        print(f"{'='*80}")
        print(f"  Plot bbox: x={x1_plot:.0f}->{x2_plot:.0f}, y={y1_plot:.0f}->{y2_plot:.0f}")
        print(f"  X-axis strip: x={x_start}->{x_end}, y={y_start}->{y_end}")
        print(f"  Strip size: {xaxis_strip.shape[1]}x{xaxis_strip.shape[0]} pixels")
        
        # Save the extracted strip
        debug_strip_path = "debug_output/xaxis_01_strip.png"
        import os
        os.makedirs("debug_output", exist_ok=True)
        cv2.imwrite(debug_strip_path, xaxis_strip)
        print(f"  Saved strip: {debug_strip_path}")
    
    # Convert to grayscale and apply threshold to find text
    gray = cv2.cvtColor(xaxis_strip, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilate to connect nearby text components (important for separated characters)
    # Use lighter dilation to avoid over-connecting separate labels
    # First: horizontal dilation to connect characters within words
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # Thinner horizontal
    dilated = cv2.dilate(binary, kernel_h, iterations=1)
    
    # Second: vertical dilation to capture full height
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 8))  # Thinner vertical
    dilated = cv2.dilate(dilated, kernel_v, iterations=1)
    
    # Third: small ellipse for slight diagonal connections
    kernel_diag = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller ellipse
    dilated = cv2.dilate(dilated, kernel_diag, iterations=1)
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STEP 2: BINARY THRESHOLDING & DILATION")
        print(f"{'='*80}")
        debug_binary_path = "debug_output/xaxis_02_binary.png"
        cv2.imwrite(debug_binary_path, binary)
        print(f"  Saved binary image: {debug_binary_path}")
        debug_dilated_path = "debug_output/xaxis_02b_dilated.png"
        cv2.imwrite(debug_dilated_path, dilated)
        print(f"  Saved dilated image: {debug_dilated_path}")
    
    # Find contours on dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"  Found {len(contours)} contours")
    
    # Get bounding boxes and filter by size
    bboxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Filter out very small noise
        if w > 5 and h > 5:
            bboxes.append((x, y, w, h, x + w // 2))  # Include center x for sorting
    
    if debug:
        print(f"  Filtered to {len(bboxes)} bboxes (removed noise < 5x5 pixels)")
        
        # Draw all detected bboxes
        debug_contours = xaxis_strip.copy()
        for i, (x, y, w, h, cx) in enumerate(bboxes):
            cv2.rectangle(debug_contours, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(debug_contours, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        debug_contours_path = "debug_output/xaxis_03_all_bboxes.png"
        cv2.imwrite(debug_contours_path, debug_contours)
        print(f"  Saved all bboxes: {debug_contours_path}")
    
    # Sort by X position
    bboxes.sort(key=lambda b: b[4])
    
    # Merge boxes based on overlap threshold
    # Only merge if they overlap by at least 50% (likely same text)
    def calculate_overlap(bbox1, bbox2):
        """Calculate the overlap percentage between two bboxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No overlap
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        smaller_area = min(bbox1_area, bbox2_area)
        
        # Return overlap as percentage of smaller bbox
        return intersection_area / smaller_area if smaller_area > 0 else 0.0
    
    merged = []
    for bbox in bboxes:
        x, y, w, h, cx = bbox
        current_bbox = (x, y, w, h)
        
        # Check if this bbox should be merged with the last merged bbox
        should_merge = False
        if merged:
            prev_x, prev_y, prev_w, prev_h = merged[-1]
            prev_bbox = (prev_x, prev_y, prev_w, prev_h)
            overlap = calculate_overlap(current_bbox, prev_bbox)
            
            # Only merge if BOTH conditions are met:
            # 1. High overlap (>= 70% to be very conservative)
            # 2. Similar Y position (same horizontal line, not tilted text stacked vertically)
            horizontal_gap = x - (prev_x + prev_w)
            vertical_gap = abs(y - prev_y)
            
            # For tilted text, bboxes will have large vertical separation
            # Only merge if on same line (vertical_gap < 10) AND high overlap
            if overlap >= 0.7 and vertical_gap < 10:
                should_merge = True
        
        if should_merge:
            # Merge with previous
            prev_x, prev_y, prev_w, prev_h = merged[-1]
            new_x = min(prev_x, x)
            new_y = min(prev_y, y)
            new_x2 = max(prev_x + prev_w, x + w)
            new_y2 = max(prev_y + prev_h, y + h)
            merged[-1] = (new_x, new_y, new_x2 - new_x, new_y2 - new_y)
            if debug:
                print(f"    Merged: Bbox at x={x} merged with previous (overlap={overlap:.2f}, v_gap={vertical_gap})")
        else:
            merged.append((x, y, w, h))
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STEP 3: BBOX MERGING (70% overlap + same line)")
        print(f"{'='*80}")
        print(f"  Merged nearby boxes: {len(bboxes)} -> {len(merged)} bboxes")
        print(f"  Note: Boxes merged only if overlap ≥ 70% AND vertical gap < 10px")
        
        # Draw merged bboxes
        debug_merged = xaxis_strip.copy()
        for i, (x, y, w, h) in enumerate(merged):
            cv2.rectangle(debug_merged, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(debug_merged, f"#{i}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        debug_merged_path = "debug_output/xaxis_04_merged_bboxes.png"
        cv2.imwrite(debug_merged_path, debug_merged)
        print(f"  Saved merged bboxes: {debug_merged_path}")
        
        for i, (x, y, w, h) in enumerate(merged):
            print(f"    Bbox #{i}: x={x}, y={y}, w={w}, h={h}")
    
    return merged, xaxis_strip, (x_start, y_start)

def ocr_xaxis_region_with_rotation(image_path, plot_bbox, debug=False):
    """
    OCR specifically the X-axis region by detecting individual text bounding boxes,
    rotating each one separately to eliminate overlap and noise.
    plot_bbox: (x1, y1, x2, y2) - the plot area coordinates
    Returns: list of word dicts with 'text' and 'bbox' in original image coordinates
    """
    img = cv2.imread(image_path)
    if img is None or plot_bbox is None:
        return []
    
    # Detect individual text regions
    text_bboxes, xaxis_strip, (x_offset, y_offset) = detect_xaxis_text_regions(image_path, plot_bbox, debug=debug)
    
    if not text_bboxes:
        if debug:
            print("\n⚠️  No text bboxes detected in X-axis region!")
        return []
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STEP 4: INDIVIDUAL BBOX OCR WITH ROTATION")
        print(f"{'='*80}")
        print(f"  Processing {len(text_bboxes)} text regions...")
    
    all_words = []
    
    # Process each detected text region individually
    for bbox_idx, (x, y, w, h) in enumerate(text_bboxes):
        # Add generous padding around the bbox (important for rotated OCR)
        # Wider horizontal padding and more upward vertical padding
        pad_x = 25  # More horizontal padding
        pad_y_top = 15  # Extra padding upward (can touch plot line)
        pad_y_bottom = 0  # Less padding downward
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y_top)
        x2 = min(xaxis_strip.shape[1], x + w + pad_x)
        y2 = min(xaxis_strip.shape[0], y + h + pad_y_bottom)
        
        text_crop = xaxis_strip[y1:y2, x1:x2]
        
        if text_crop.size == 0:
            continue
        
        crop_h, crop_w = text_crop.shape[:2]
        
        if debug:
            print(f"\n  --- Bbox #{bbox_idx} ---")
            print(f"      Position: x={x}, y={y}, w={w}, h={h}")
            print(f"      Crop size: {crop_w}x{crop_h}")
            debug_crop_path = f"debug_output/xaxis_05_bbox{bbox_idx:02d}_crop.png"
            cv2.imwrite(debug_crop_path, text_crop)
            print(f"      Saved crop: {debug_crop_path}")
        
        best_text = None
        best_conf = 0
        best_bbox_full = None
        best_orientation = None
        
        # Helper function to rotate image by arbitrary angle
        def rotate_image(image, angle):
            """Rotate image by arbitrary angle (positive = CCW)"""
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            # Calculate new dimensions to avoid clipping
            cos = abs(M[0, 0])
            sin = abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]
            rotated = cv2.warpAffine(image, M, (new_w, new_h), 
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))
            return rotated
        
        # Check if text is likely already horizontal (width > height)
        # If so, try normal orientation first and only rotate if confidence is low
        is_likely_horizontal = crop_w > crop_h * 1.2
        
        # Try multiple orientations on this individual crop
        if is_likely_horizontal:
            # Try normal first, only rotate if confidence is poor
            orientations_to_try = [
                ("normal", text_crop, 0),
            ]
            try_rotations = False  # Will be set to True if normal has low confidence
        else:
            # Text is vertical/tilted, try all orientations including 45° angles
            orientations_to_try = [
                ("normal", text_crop, 0),
                ("45_ccw", rotate_image(text_crop, 45), 45),
                ("90_ccw", cv2.rotate(text_crop, cv2.ROTATE_90_COUNTERCLOCKWISE), 90),
                ("45_cw", rotate_image(text_crop, -45), -45),
                ("90_cw", cv2.rotate(text_crop, cv2.ROTATE_90_CLOCKWISE), -90),
            ]
            try_rotations = False
        
        if debug:
            if is_likely_horizontal:
                print(f"      Text appears horizontal (w={crop_w} > h={crop_h}), trying normal first...")
            else:
                print(f"      Text appears tilted/vertical (w={crop_w} ≤ h={crop_h}), trying {len(orientations_to_try)} orientations (0°, ±45°, ±90°)...")
        
        for orientation_name, rotated_crop, angle in orientations_to_try:
            try:
                if debug:
                    # Save rotated version
                    debug_rot_path = f"debug_output/xaxis_05_bbox{bbox_idx:02d}_{orientation_name}.png"
                    cv2.imwrite(debug_rot_path, rotated_crop)
                
                rgb = cv2.cvtColor(rotated_crop, cv2.COLOR_BGR2RGB)
                pil_crop = Image.fromarray(rgb)
                
                # OCR with confidence scores
                data = pytesseract.image_to_data(pil_crop, lang="eng", output_type=pytesseract.Output.DICT)
                
                found_texts = []
                for i, txt in enumerate(data["text"]):
                    txt = (txt or "").strip()
                    if not txt or len(txt) < 2:
                        continue
                    
                    conf = int(data["conf"][i]) if data["conf"][i] != -1 else 0
                    found_texts.append((txt, conf))
                    
                    # Pick the result with highest confidence
                    if conf > best_conf:
                        best_conf = conf
                        best_text = txt
                        best_orientation = orientation_name
                        
                        # Calculate bbox in original image coordinates
                        if angle == 0:  # Normal
                            best_bbox_full = [x_offset + x1, y_offset + y1, x_offset + x2, y_offset + y2]
                        else:  # Rotated - use the detected text region position
                            best_bbox_full = [x_offset + x, y_offset + y, x_offset + x + w, y_offset + y + h]
                
                if debug and found_texts:
                    print(f"        {orientation_name:10s}: {found_texts}")
            except Exception as e:
                if debug:
                    print(f"        {orientation_name:10s}: ERROR - {e}")
                continue
        
        # If text appeared horizontal but confidence is low, try rotations as fallback
        if is_likely_horizontal and best_conf < 50 and not try_rotations:
            if debug:
                print(f"      Low confidence ({best_conf}), trying rotations as fallback...")
            
            # Add rotated orientations and retry (including 45° for tilted text)
            rotated_orientations = [
                ("45_ccw", rotate_image(text_crop, 45), 45),
                ("90_ccw", cv2.rotate(text_crop, cv2.ROTATE_90_COUNTERCLOCKWISE), 90),
                ("45_cw", rotate_image(text_crop, -45), -45),
                ("90_cw", cv2.rotate(text_crop, cv2.ROTATE_90_CLOCKWISE), -90),
            ]
            
            for orientation_name, rotated_crop, angle in rotated_orientations:
                try:
                    if debug:
                        debug_rot_path = f"debug_output/xaxis_05_bbox{bbox_idx:02d}_{orientation_name}.png"
                        cv2.imwrite(debug_rot_path, rotated_crop)
                    
                    rgb = cv2.cvtColor(rotated_crop, cv2.COLOR_BGR2RGB)
                    pil_crop = Image.fromarray(rgb)
                    data = pytesseract.image_to_data(pil_crop, lang="eng", output_type=pytesseract.Output.DICT)
                    
                    found_texts = []
                    for i, txt in enumerate(data["text"]):
                        txt = (txt or "").strip()
                        if not txt or len(txt) < 2:
                            continue
                        
                        conf = int(data["conf"][i]) if data["conf"][i] != -1 else 0
                        found_texts.append((txt, conf))
                        
                        if conf > best_conf:
                            best_conf = conf
                            best_text = txt
                            best_orientation = orientation_name
                            best_bbox_full = [x_offset + x, y_offset + y, x_offset + x + w, y_offset + y + h]
                    
                    if debug and found_texts:
                        print(f"        {orientation_name:10s}: {found_texts}")
                except Exception as e:
                    if debug:
                        print(f"        {orientation_name:10s}: ERROR - {e}")
                    continue
        
        # Add the best result for this bbox
        if best_text and best_bbox_full:
            all_words.append({
                "text": best_text,
                "bbox": best_bbox_full
            })
            if debug:
                print(f"      ✓ BEST: '{best_text}' (conf: {best_conf}, orientation: {best_orientation})")
        else:
            if debug:
                print(f"      ✗ No text detected in any orientation")
    
    if debug:
        print(f"\n{'='*80}")
        print(f"STEP 5: FINAL RESULTS")
        print(f"{'='*80}")
        print(f"  Detected {len(all_words)} X-axis labels:")
        for i, word in enumerate(all_words):
            print(f"    [{i}] '{word['text']}' at bbox {word['bbox']}")
        print(f"{'='*80}\n")
    
    return all_words

def extract_xaxis_labels(word_infos, cls_info, img_height, image_path=None):
    """
    Extract X-axis labels (category labels below the plot area).
    Handles horizontal, tilted, and vertical labels.
    """
    if 5 not in cls_info:
        return []
    
    plot = _to_box(cls_info[5])
    if plot is None:
        return []
    
    x1_plot, y1_plot, x2_plot, y2_plot = plot
    
    # If image_path provided, also do rotation OCR specifically for X-axis region
    extra_words = []
    if image_path:
        print(f"\n[X-axis OCR] Using plot coordinates: {plot}")
        extra_words = ocr_xaxis_region_with_rotation(image_path, plot, debug=True)
        # Merge extra words into word_infos
        all_word_infos = list(word_infos) + extra_words
    else:
        all_word_infos = word_infos
    
    # Look for text below the plot area
    candidate_words = []
    for w in all_word_infos:
        x_center = (w["bbox"][0] + w["bbox"][2]) / 2
        y_center = (w["bbox"][1] + w["bbox"][3]) / 2
        bbox_width = w["bbox"][2] - w["bbox"][0]
        bbox_height = w["bbox"][3] - w["bbox"][1]
        
        # Check if text is below plot area and horizontally within it
        text = w["text"].strip()
        
        # Exclude only percentage signs with decimals (like "14.5%")
        # Allow years (4-digit numbers) and other integers as they are likely X-axis labels
        is_percentage = '%' in text and '.' in text  # Only exclude decimal percentages
        
        # For vertical/tilted text, height will be > width
        # Relax the position constraints for vertical labels
        is_likely_vertical = bbox_height > bbox_width * 1.5
        
        # Extended search area for vertical labels
        vertical_tolerance = 100 if is_likely_vertical else 0
        
        if (y_center > y2_plot - vertical_tolerance and 
            x_center >= x1_plot - 20 and  # Allow slight overhang
            x_center <= x2_plot + 20 and
            len(text) >= 2 and
            not is_percentage):  # Not a percentage label
            candidate_words.append({
                "text": text,
                "x_center": x_center,
                "x_left": w["bbox"][0],
                "x_right": w["bbox"][2],
                "y_pos": y_center,
                "y_top": w["bbox"][1],
                "y_bottom": w["bbox"][3],
                "bbox": w["bbox"],
                "is_vertical": is_likely_vertical
            })
    
    # Find the main X-axis label row (typically the one closest to plot but not too far)
    # Group by similar Y positions to find rows
    if not candidate_words:
        return []
    
    # Sort by Y position to find rows
    candidate_words.sort(key=lambda x: x["y_top"])
    
    # Find the first main row of labels (should be closest to plot area)
    main_row_words = []
    if candidate_words:
        first_y = candidate_words[0]["y_top"]
        # Get all words in the first row (within 20 pixels vertically)
        main_row_words = [w for w in candidate_words if abs(w["y_top"] - first_y) < 20]
    
    # Sort main row by X position
    main_row_words.sort(key=lambda x: x["x_left"])
    
    # Debug: Show detected X-axis candidates
    if main_row_words:
        print(f"  X-axis label candidates: {[w['text'] for w in main_row_words]}")
    
    # Convert each detected word into a label (don't group them)
    # Since we're extracting individual rotated labels, each one is already separate
    final_labels = []
    for word in main_row_words:
        final_labels.append({
            "text": word["text"],
            "x_center": word["x_center"],
            "y_pos": word["y_pos"],
            "bbox": word["bbox"]
        })
    
    # Sort by X position (left to right)
    final_labels.sort(key=lambda x: x["x_center"])
    
    # Debug: Show what we're returning
    print(f"\n[extract_xaxis_labels] Returning {len(final_labels)} labels:")
    for i, label in enumerate(final_labels[:10]):  # Show first 10
        print(f"  [{i}] '{label['text']}' at x={label['x_center']:.1f}")
    if len(final_labels) > 10:
        print(f"  ... and {len(final_labels) - 10} more")
    
    return final_labels

def map_labels_to_bars(bars, x_labels):
    """Map X-axis labels to bars based on horizontal alignment"""
    if not x_labels or not bars:
        return {}
    
    bar_label_map = {}
    
    for i, (x1, y1, x2, y2) in enumerate(bars):
        bar_center = (x1 + x2) / 2
        
        # Find closest label
        min_dist = float('inf')
        closest_label = None
        
        for label in x_labels:
            dist = abs(bar_center - label["x_center"])
            if dist < min_dist:
                min_dist = dist
                closest_label = label["text"]
        
        # Only assign if reasonably close (within reasonable distance)
        if min_dist < 100:  # Adjust threshold as needed
            bar_label_map[i] = closest_label
    
    return bar_label_map

def map_labels_to_line_points(line_points, x_labels, plot_bounds=None):
    """
    Map X-axis labels to line chart points based on horizontal alignment.
    Returns dict mapping point index to X-axis label.
    """
    if not x_labels or not line_points:
        return {}
    
    point_label_map = {}
    
    # Get plot X boundaries if available to normalize positions
    x_min, x_max = None, None
    if plot_bounds:
        x_min = plot_bounds[0]
        x_max = plot_bounds[2]
    
    # Create bins/regions for each X-axis label
    # Assume labels are evenly spaced or use their actual positions
    if len(x_labels) > 1:
        # Sort labels by X position
        sorted_labels = sorted(x_labels, key=lambda l: l["x_center"])
        
        # For each point, find the nearest X-axis label
        for i, point in enumerate(line_points):
            if isinstance(point, dict):
                bbox = point.get('bbox', [0, 0, 0, 0])
                point_x = bbox[0] + bbox[2] / 2.0
                
                # Find closest label
                min_dist = float('inf')
                closest_label = None
                
                for label in sorted_labels:
                    dist = abs(point_x - label["x_center"])
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = label["text"]
                
                # Assign if reasonably close
                if min_dist < 80:  # Threshold for line charts (can be tighter)
                    point_label_map[i] = closest_label
    
    return point_label_map


# ------------------------------------------------------------------
# Legend / color matching
# ------------------------------------------------------------------

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in rgb])

def extract_dominant_color(img, box, use_median=True):
    """
    Extract dominant color from a region using median (more robust to outliers)
    or mode (most common color). Also filters out white/gray background pixels.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: 
        return (0, 0, 0)
    
    # Use center 60% to avoid edges/borders
    h, w = roi.shape[:2]
    ys, ye = int(h * 0.2), int(h * 0.8)
    xs, xe = int(w * 0.2), int(w * 0.8)
    core = roi[ys:ye, xs:xe]
    
    if core.size == 0:
        core = roi
    
    # Reshape to list of pixels
    pixels = core.reshape(-1, 3).astype(np.float32)
    
    # Filter out near-white pixels (background) - BGR format
    # Keep pixels where at least one channel is significantly different from others
    mask = np.any(np.abs(pixels - pixels.mean(axis=1, keepdims=True)) > 30, axis=1)
    
    # Also filter out very light pixels (likely background)
    brightness_mask = pixels.mean(axis=1) < 240
    mask = mask & brightness_mask
    
    if mask.sum() > 0:
        filtered_pixels = pixels[mask]
    else:
        filtered_pixels = pixels
    
    if use_median:
        # Median is more robust to outliers
        bgr = np.median(filtered_pixels, axis=0)
    else:
        # Mean color
        bgr = np.mean(filtered_pixels, axis=0)
    
    # Convert BGR to RGB
    return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def bar_color(img, box):
    """Extract color from bar chart bar"""
    return extract_dominant_color(img, box, use_median=True)

def point_color(img, x, y, radius=4):
    """
    Extract color from a point on a line chart.
    Uses a focused sample to get accurate line color.
    """
    H, W = img.shape[:2]
    x, y = int(x), int(y)
    
    # Get small region directly at the point
    x1 = max(0, x - radius)
    y1 = max(0, y - radius)
    x2 = min(W - 1, x + radius)
    y2 = min(H - 1, y + radius)
    
    roi = img[y1:y2, x1:x2]
    if roi.size == 0:
        return (0, 0, 0)
    
    # Get all pixels in the small region
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    # Filter out very bright pixels (white background/grid)
    brightness = pixels.mean(axis=1)
    mask = brightness < 250
    
    if mask.sum() > 0:
        filtered_pixels = pixels[mask]
        # Use median to avoid outliers
        bgr = np.median(filtered_pixels, axis=0)
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))
    else:
        # If all pixels are bright, just use median of all
        bgr = np.median(pixels, axis=0)
        return (int(bgr[2]), int(bgr[1]), int(bgr[0]))

def find_legend_pairs(img, words):
    """
    Find legend items by looking for text labels with colored markers.
    Improved to search multiple regions and use better color extraction.
    """
    H, W = img.shape[:2]
    legend = []
    
    # Search in right side (most common) and top/bottom regions
    search_regions = [
        ("right", lambda x1, y1: x1 >= int(W * 0.65)),  # Right 35%
        ("top-right", lambda x1, y1: x1 >= int(W * 0.50) and y1 <= int(H * 0.20)),  # Top-right
        ("bottom", lambda x1, y1: y1 >= int(H * 0.85)),  # Bottom 15%
    ]
    
    seen_texts = set()
    
    for region_name, condition in search_regions:
        for w in words:
            x1, y1, x2, y2 = w["bbox"]
            text = w["text"].strip()
            
            # Skip short text, numbers, and already seen labels
            if len(text) < 2 or text in seen_texts:
                continue
            if text.replace('.', '').replace(',', '').replace('%', '').isdigit():
                continue
            
            if not condition(x1, y1):
                continue
            
            # Look for color marker to the left of text
            marker_found = False
            for offset in [12, 18, 24, 30]:  # Try multiple distances
                cx = max(0, x1 - offset)
                cy = int((y1 + y2) / 2)
                
                # Sample a small square around the marker position
                sx1, sy1 = max(0, cx - 6), max(0, cy - 6)
                sx2, sy2 = min(W - 1, cx + 6), min(H - 1, cy + 6)
                patch = img[sy1:sy2, sx1:sx2]
                
                if patch.size == 0:
                    continue
                
                # Extract color using improved method
                rgb = extract_dominant_color(img, (sx1, sy1, sx2, sy2), use_median=True)
                
                # Check if this is a valid color (not white/gray background)
                r, g, b = rgb
                brightness = (r + g + b) / 3
                variance = max(abs(r - g), abs(g - b), abs(r - b))
                
                # Valid colored marker: not too bright and has color variance
                if brightness < 220 and (variance > 20 or brightness < 150):
                    legend.append({
                        "text": text,
                        "rgb": rgb,
                        "hex": rgb_to_hex(rgb),
                        "region": region_name
                    })
                    seen_texts.add(text)
                    marker_found = True
                    break
            
            if marker_found:
                continue
    
    return legend

def closest_legend(rgb, legend, max_distance=10000):
    """
    Find closest legend item by color with improved matching.
    Uses Euclidean distance in RGB space with threshold.
    """
    if not legend:
        return None
    
    r1, g1, b1 = rgb
    best, best_d = None, float('inf')
    
    for item in legend:
        r2, g2, b2 = item["rgb"]
        
        # Euclidean distance in RGB space
        distance = (r1 - r2)**2 + (g1 - g2)**2 + (b1 - b2)**2
        
        if distance < best_d:
            best, best_d = item, distance
    
    # Only return match if distance is reasonable (not too far apart)
    if best_d < max_distance:
        return best
    
    return None


# ------------------------------------------------------------------
# Main per-image pipeline
# ------------------------------------------------------------------

def run_on_image(image_path, chart_type="Bar", save_path=None, methods_override=None, return_images=False):
    model_bundle = methods_override or methods
    if model_bundle is None:
        raise RuntimeError("Models not loaded. Call Pre_load_nets first.")

    pil_img = Image.open(image_path).convert("RGB")
    original_b64 = _encode_image_to_base64(pil_img) if return_images else None

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read {image_path}")

    overlay_b64 = original_b64 if return_images else None

    with torch.no_grad():
        cls_db, cls_net, cls_fn = model_bundle["Cls"]
        cls_res = cls_fn(img, cls_db, cls_net, debug=False)
        tls, brs = cls_res[1], cls_res[2]
        _, raw_info = GroupCls(pil_img, tls, brs)
        cls_info = {int(k): _to_box(v) for k,v in raw_info.items()}

        words = ocr_result_full_image(image_path)
        titles = extract_titles_from_clsinfo(cls_info, words)
        y_min, y_max = extract_axis_scale_from_clsinfo(img, cls_info)

        bars_raw = []
        lines_raw = []
        pie_segments = []
        
        if chart_type == "Bar" and "Bar" in model_bundle:
            bdb, bnet, bfn = model_bundle["Bar"]
            bres = bfn(img, bdb, bnet, debug=False)
            tls_b, brs_b = bres[0], bres[1]
            # Convert cv2 image to PIL for GroupBarRaw (it expects PIL Image)
            pil_img_copy = pil_img.copy() if hasattr(pil_img, 'copy') else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bars_raw = GroupBarRaw(pil_img_copy, tls_b, brs_b)
            if return_images:
                overlay_b64 = _encode_image_to_base64(pil_img_copy)
        
        elif chart_type == "Line" and "Line" in model_bundle:
            ldb, lnet, lfn = model_bundle["Line"]
            lres = lfn(img, ldb, lnet, debug=False)
            keys_raw, hybrids_raw = lres[0], lres[1]
            # Convert cv2 image to PIL for GroupQuiryRaw
            pil_img_copy = pil_img.copy() if hasattr(pil_img, 'copy') else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # GroupQuiryRaw returns (image, quiries, keys, hybrids)
            result_line = GroupQuiryRaw(pil_img_copy, keys_raw, hybrids_raw)
            if result_line and len(result_line) == 4:
                pil_img_copy, quiries, line_keys, line_hybrids = result_line
                lines_raw = line_keys  # Extract the key points from lines
            else:
                lines_raw = []
            if return_images:
                overlay_b64 = _encode_image_to_base64(pil_img_copy)
        
        elif chart_type == "Pie" and "Pie" in model_bundle:
            pdb, pnet, pfn = model_bundle["Pie"]
            pres = pfn(img, pdb, pnet, debug=False)
            tls_p, brs_p = pres[0], pres[1]
            # Convert cv2 image to PIL for GroupPie
            pil_img_copy = pil_img.copy() if hasattr(pil_img, 'copy') else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # GroupPie returns (annotated_image, list_of_angles)
            result_pie = GroupPie(pil_img_copy, tls_p, brs_p)
            if result_pie and len(result_pie) == 2:
                pil_img_copy, pie_angles = result_pie
                pie_segments = pie_angles  # List of angles (in degrees)
            else:
                pie_segments = []
            if return_images:
                overlay_b64 = _encode_image_to_base64(pil_img_copy)

        # Extract X-axis labels (category labels) - mainly for bar/line charts
        x_axis_labels = extract_xaxis_labels(words, cls_info, img.shape[0], image_path)
        bar_x_labels = map_labels_to_bars(bars_raw, x_axis_labels) if chart_type == "Bar" else {}
        
        legend_items = find_legend_pairs(img, words)

        # Fit pixel→value regression - improved version with better Y-axis detection
        tick_pairs = []
        plot_region = None
        if 5 in cls_info:
            plot_region = cls_info[5]  # [x1, y1, x2, y2] of plot area
            x1_plot, y1_plot, x2_plot, y2_plot = plot_region
            
            # Strategy 1: Try to find Y-axis labels in full image first (more reliable)
            # Look for numbers on the far left side of the image
            potential_ticks = []
            for w in words:
                # Be very strict: must be on far left (X < 60 pixels from left edge)
                if w["bbox"][0] < 60:
                    txt = w["text"].strip()
                    # Must be a simple number (not complex text)
                    # Allow single digits, two digits, decimals with comma or dot
                    if re.match(r'^\d{1,2}([,.]\d{1,2})?$', txt):
                        txt_clean = txt.replace(',', '.')
                        try:
                            val = float(txt_clean)
                            # Reject unreasonably large values (likely OCR errors)
                            # Most bar charts have Y-axis max under 100
                            if val <= 100:
                                y_pos = (w["bbox"][1] + w["bbox"][3]) / 2.0
                                potential_ticks.append((y_pos, val))
                        except:
                            pass
            
            # If Strategy 1 found ticks, use them
            if len(potential_ticks) >= 2:
                tick_pairs = potential_ticks
            
            # Sort by Y-position (top to bottom) - this will be done later after all strategies
            # Smart filtering: Use top and bottom ticks as anchors, infer the rest
            # This filtering will be applied AFTER all strategies complete
            if False:  # Placeholder - filtering moved to end
                print(f"\nDEBUG: Filtering {len(potential_ticks)} potential ticks using anchor method...")
                
                # Get top tick (smallest Y, highest value in normal bar chart)
                top_tick = potential_ticks[0]
                # Get bottom tick (largest Y, lowest value - usually 0)
                bottom_tick = potential_ticks[-1]
                
                print(f"  Top anchor: Y={top_tick[0]:.1f}, Value={top_tick[1]}")
                print(f"  Bottom anchor: Y={bottom_tick[0]:.1f}, Value={bottom_tick[1]}")
                
                # Calculate the value range and Y-pixel range
                value_range = abs(top_tick[1] - bottom_tick[1])
                y_pixel_range = abs(bottom_tick[0] - top_tick[0])
                
                # Infer number of ticks from vertical spacing
                # Typical tick spacing is 25-35 pixels
                if y_pixel_range > 0:
                    avg_y_spacing = y_pixel_range / (len(potential_ticks) - 1) if len(potential_ticks) > 1 else y_pixel_range
                    print(f"  Average Y spacing: {avg_y_spacing:.1f} pixels")
                    print(f"  Value range: {value_range}")
                    
                    # Determine expected number of intervals
                    # This should match the value range if ticks are at integer intervals
                    # For example: 0,1,2,3,4,5,6,7,8 = 9 ticks = 8 intervals, range = 8
                    expected_intervals = value_range
                    expected_num_ticks = int(expected_intervals) + 1
                    
                    print(f"  Expected number of ticks: {expected_num_ticks} (based on range)")
                    
                    # If we have close to the expected number, validate middle ticks
                    if abs(len(potential_ticks) - expected_num_ticks) <= 2:
                        print(f"  Tick count matches expectation, validating middle values...")
                        
                        # Calculate value per interval
                        value_per_interval = value_range / expected_intervals if expected_intervals > 0 else 0
                        
                        # Validate each tick
                        corrected_ticks = [top_tick]
                        for i in range(1, len(potential_ticks) - 1):
                            y_pos, detected_val = potential_ticks[i]
                            
                            # Calculate expected value based on Y position
                            y_progress = (y_pos - top_tick[0]) / y_pixel_range  # 0 to 1
                            expected_val = top_tick[1] - (y_progress * value_range)  # Assuming descending
                            
                            # Check if detected value is reasonable
                            error = abs(detected_val - expected_val)
                            error_pct = error / max(value_per_interval, 0.1)
                            
                            print(f"    Tick {i}: Y={y_pos:.1f}, detected={detected_val}, expected={expected_val:.1f}, error={error_pct:.1%}")
                            
                            # If error is large (>40%), use corrected value
                            if error_pct > 0.4:
                                # Round to nearest interval
                                corrected_val = round(expected_val / value_per_interval) * value_per_interval
                                print(f"      -> CORRECTING: {detected_val} → {corrected_val}")
                                corrected_ticks.append((y_pos, corrected_val))
                            else:
                                corrected_ticks.append((y_pos, detected_val))
                        
                        corrected_ticks.append(bottom_tick)
                        tick_pairs = corrected_ticks
                        print(f"  Final: {len(tick_pairs)} ticks after validation")
                    else:
                        # Number of ticks doesn't match - just use top and bottom as anchors
                        print(f"  Tick count mismatch, using only top and bottom anchors")
                        tick_pairs = [top_tick, bottom_tick]
                else:
                    tick_pairs = potential_ticks
            else:
                tick_pairs = potential_ticks
            
            # Strategy 2: Look for percentage labels on the bars themselves  
            # Many charts show values like "14,6%" on top of bars
            if len(tick_pairs) < 2:
                bar_labels = []
                for w in words:
                    txt = w["text"]
                    # Look for percentage patterns
                    if '%' in txt or ',' in txt:
                        # Extract number from patterns like "14,6%" or "22,2%"
                        match = re.search(r'(\d+)[,.]?(\d*)\s*%?', txt)
                        if match:
                            try:
                                if match.group(2):
                                    val = float(match.group(1) + '.' + match.group(2))
                                else:
                                    val = float(match.group(1))
                                y_pos = (w["bbox"][1] + w["bbox"][3]) / 2.0
                                bar_labels.append((y_pos, val))
                            except:
                                pass
                
                # If we found bar labels, use them to infer Y-axis scale
                if len(bar_labels) >= 2:
                    tick_pairs = bar_labels
            
            # Strategy 3: If still no ticks, OCR left margin with aggressive settings
            if len(tick_pairs) < 2:
                tick_pairs = []
                # Try the entire left margin of the image (not just plot area)
                # Extend the bottom to catch "0" which might be at the very bottom edge
                left_margin = img[0:min(img.shape[0], int(y2_plot) + 50), 0:min(120, int(x1_plot))]
                if left_margin.size > 0:
                    # Try multiple OCR strategies
                    left_upscaled = cv2.resize(left_margin, (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                    
                    # Try PSM 6 (uniform block of text)
                    for psm_mode in [6, 11, 13]:
                        data = pytesseract.image_to_data(
                            Image.fromarray(cv2.cvtColor(left_upscaled, cv2.COLOR_BGR2RGB)),
                            config=f"--psm {psm_mode} -c tessedit_char_whitelist=0123456789",
                            output_type=pytesseract.Output.DICT)
                        for i, txt in enumerate(data["text"]):
                            txt_clean = re.sub(r'[^0-9]', '', (txt or "").strip())
                            if txt_clean and len(txt_clean) >= 1:
                                try:
                                    val = float(txt_clean)
                                    # Map back to full image Y coordinate
                                    strip_y = data["top"][i] / 3.0  # Downscale from 3x
                                    full_img_y = strip_y + (data["height"][i] / 6.0)
                                    tick_pairs.append((full_img_y, val))
                                except:
                                    pass
                        if len(tick_pairs) >= 2:
                            break
            
            # Post-process: If we're missing a "0" tick at the bottom, add it
            # This is common in bar charts where 0 is the baseline
            if len(tick_pairs) >= 2:
                tick_pairs.sort(key=lambda x: x[0])
                bottom_value = tick_pairs[-1][1]
                
                # If the bottom value is not 0 or close to 0, check if we should add a 0 tick
                if bottom_value > 0.5:  # Not already close to 0
                    # Check if there's a "0" at the very bottom that we might have missed
                    # Look in the bottom region of the left margin
                    bottom_region = img[int(y2_plot):min(img.shape[0], int(y2_plot) + 40), 0:min(80, int(x1_plot))]
                    if bottom_region.size > 0:
                        bottom_upscaled = cv2.resize(bottom_region, (0,0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
                        try:
                            bottom_text = pytesseract.image_to_string(
                                Image.fromarray(cv2.cvtColor(bottom_upscaled, cv2.COLOR_BGR2RGB)),
                                config="--psm 8 -c tessedit_char_whitelist=0"
                            ).strip()
                            if '0' in bottom_text:
                                # Add a 0 tick at the estimated position
                                # Use linear extrapolation from existing ticks
                                if len(tick_pairs) >= 2:
                                    # Calculate pixel spacing per value unit
                                    y_per_val = (tick_pairs[-1][0] - tick_pairs[0][0]) / (tick_pairs[0][1] - tick_pairs[-1][1])
                                    # Extrapolate to where 0 should be
                                    zero_y = tick_pairs[-1][0] + (tick_pairs[-1][1] - 0) * y_per_val
                                    # Make sure it's within reasonable bounds
                                    if zero_y <= img.shape[0] and zero_y > tick_pairs[-1][0]:
                                        tick_pairs.append((zero_y, 0.0))
                        except:
                            pass
        
        # NOW apply smart filtering to all collected ticks (from any strategy)
        # Use top and bottom ticks as anchors and validate/correct the middle ones
        if len(tick_pairs) >= 2:
            # Sort by Y-position (top to bottom)
            tick_pairs.sort(key=lambda x: x[0])
            
            # Get top tick (smallest Y, highest value in normal bar chart)
            top_tick = tick_pairs[0]
            # Get bottom tick (largest Y, lowest value - usually 0)
            bottom_tick = tick_pairs[-1]
            
            # Check if we should add a 0 tick at the bottom
            # Most bar charts have 0 as the baseline
            if bottom_tick[1] > 0.5 and len(tick_pairs) >= 2:
                # Calculate the expected Y position for 0 using linear interpolation
                # from the existing ticks
                y_per_value = (bottom_tick[0] - top_tick[0]) / (top_tick[1] - bottom_tick[1])
                zero_y = bottom_tick[0] + (bottom_tick[1] - 0) * y_per_value
                
                # Only add if it's reasonable (within plot bounds + small margin)
                if plot_region is not None:
                    _, _, _, y2_plot = plot_region
                    # Allow up to 60 pixels below the plot area for the 0 tick
                    if zero_y <= y2_plot + 60 and zero_y > bottom_tick[0]:
                        tick_pairs.append((zero_y, 0.0))
                        tick_pairs.sort(key=lambda x: x[0])
                        bottom_tick = tick_pairs[-1]  # Update bottom tick
            
            # Recalculate with potentially updated bottom tick
            top_tick = tick_pairs[0]
            bottom_tick = tick_pairs[-1]
            
            # Calculate the value range and Y-pixel range
            value_range = abs(top_tick[1] - bottom_tick[1])
            y_pixel_range = abs(bottom_tick[0] - top_tick[0])
            
            if y_pixel_range > 0 and value_range > 0:
                # Determine expected number of intervals based on value range
                # For example: 0,1,2,3,4,5,6,7,8 = 9 ticks = 8 intervals, range = 8
                expected_intervals = value_range
                expected_num_ticks = int(expected_intervals) + 1
                
                # If we have close to the expected number, validate and correct middle ticks
                if abs(len(tick_pairs) - expected_num_ticks) <= 3:
                    # Calculate value per interval
                    value_per_interval = value_range / expected_intervals
                    
                    # Validate and correct each tick
                    corrected_ticks = [top_tick]
                    for i in range(1, len(tick_pairs) - 1):
                        y_pos, detected_val = tick_pairs[i]
                        
                        # Calculate expected value based on Y position
                        y_progress = (y_pos - top_tick[0]) / y_pixel_range  # 0 to 1
                        expected_val = top_tick[1] - (y_progress * value_range)  # Descending (top=high, bottom=low)
                        
                        # Check if detected value is reasonable
                        error = abs(detected_val - expected_val)
                        error_pct = error / max(value_per_interval, 0.1)
                        
                        # If error is large (>40% of one interval), use corrected value
                        if error_pct > 0.4:
                            # Round to nearest interval
                            corrected_val = round(expected_val / value_per_interval) * value_per_interval
                            corrected_ticks.append((y_pos, corrected_val))
                        else:
                            corrected_ticks.append((y_pos, detected_val))
                    
                    corrected_ticks.append(bottom_tick)
                    tick_pairs = corrected_ticks
                else:
                    # Number of ticks doesn't match - just use top and bottom as reliable anchors
                    tick_pairs = [top_tick, bottom_tick]
        
        # Print detected Y-axis ticks for debugging
        if tick_pairs:
            print(f'\nDetected Y-axis ticks: {len(tick_pairs)} ticks found')
            print('  Y-pixel  →  Value')
            print('  ' + '-' * 25)
            for y_pos, val in sorted(tick_pairs, key=lambda x: x[0]):
                print(f'  {y_pos:7.1f}  →  {val:8.2f}')
            print()
        else:
            print('\nWarning: No Y-axis ticks detected!')
            print('Falling back to estimated Y-axis range from plot region.\n')
        
        # Determine value mapping function
        # In image coordinates: smaller Y pixel = top of image = higher values
        # larger Y pixel = bottom of image = lower values
        a = b = None
        
        if len(tick_pairs) >= 2:
            # Use linear regression from detected ticks
            ys = np.array([p[0] for p in tick_pairs])
            vs = np.array([p[1] for p in tick_pairs])
            A = np.vstack([ys, np.ones_like(ys)]).T
            a, b = np.linalg.lstsq(A, vs, rcond=None)[0]
            # Update y_min and y_max based on detected ticks for accurate display
            y_min = float(vs.min())
            y_max = float(vs.max())
        elif y_min is not None and y_max is not None and plot_region is not None:
            # Fallback: use estimated min/max and plot region bounds
            # In a bar chart, y_max is at top (smaller y pixel), y_min at bottom (larger y pixel)
            x1_plot, y1_plot, x2_plot, y2_plot = plot_region
            # Linear mapping: value = a * y_pixel + b
            # At y1_plot (top, smaller Y): value = y_max
            # At y2_plot (bottom, larger Y): value = y_min
            # Slope is negative: as Y increases (go down), value decreases
            a = (y_min - y_max) / (y2_plot - y1_plot)
            b = y_max - a * y1_plot
        
        def y_to_val(y):
            if a is not None and b is not None:
                return float(a * y + b)
            return None
        
        # Warn if Y-axis calibration failed for line/bar charts
        if chart_type in ["Line", "Bar"] and a is None:
            print(f"\n⚠️  WARNING: Y-axis calibration failed for {chart_type} chart!")
            print(f"   - No Y-axis tick labels detected")
            print(f"   - Y-values will be shown as N/A")
            print(f"   - Check if Y-axis labels are visible and clear in the image\n")

        rows = []
        
        # Process Pie chart data
        if chart_type == "Pie" and pie_segments:
            # pie_segments contains angles in degrees for each segment
            # Calculate percentages from angles
            # For pie charts, try to extract colors from the center region
            H, W = img.shape[:2]
            center_x, center_y = W // 2, H // 2
            
            for i, angle in enumerate(pie_segments):
                percentage = (angle / 360.0) * 100.0 if angle > 0 else 0.0
                
                # Try to extract color from pie segment
                # Estimate angle position (this is approximate)
                angle_offset = sum(pie_segments[:i]) if i > 0 else 0
                mid_angle = angle_offset + angle / 2.0
                
                # Convert to radians and get point on the pie segment
                angle_rad = np.radians(mid_angle - 90)  # -90 to start from top
                radius = min(W, H) * 0.25  # Estimate pie radius
                
                sample_x = int(center_x + radius * np.cos(angle_rad))
                sample_y = int(center_y + radius * np.sin(angle_rad))
                
                # Extract color from this point
                rgb = point_color(img, sample_x, sample_y, radius=8)
                match = closest_legend(rgb, legend_items)
                
                rows.append({
                    "segment_index": i + 1,
                    "category": match["text"] if match else "",
                    "label": match["text"] if match else "",
                    "color": match["hex"] if match else rgb_to_hex(rgb),
                    "angle_degrees": round(angle, 2),
                    "value": round(percentage, 2),
                })
        
        # Process Line chart data
        elif chart_type == "Line" and lines_raw:
            # Map line points to X-axis labels
            line_x_labels = map_labels_to_line_points(lines_raw, x_axis_labels, plot_region)
            
            # First pass: collect all points by series to sample colors better
            series_points = {}
            for i, point in enumerate(lines_raw):
                if isinstance(point, dict):
                    category_id = point.get('category_id', 0)
                    if category_id not in series_points:
                        series_points[category_id] = []
                    series_points[category_id].append((i, point))
            
            # Extract colors by sampling from each series
            series_colors = {}
            for category_id, points in series_points.items():
                # Sample from middle point of the series for most accurate color
                mid_idx = len(points) // 2
                _, mid_point = points[mid_idx]
                bbox = mid_point.get('bbox', [0, 0, 0, 0])
                x_pixel = bbox[0] + bbox[2] / 2.0
                y_pixel = bbox[1] + bbox[3] / 2.0
                
                # Get color from this representative point
                rgb = point_color(img, x_pixel, y_pixel, radius=4)
                match = closest_legend(rgb, legend_items)
                
                series_colors[category_id] = {
                    'rgb': rgb,
                    'hex': rgb_to_hex(rgb),
                    'label': match["text"] if match else f"Series {category_id}"
                }
                # Debug: print detected series color
                print(f"  Series {category_id}: Color {rgb_to_hex(rgb)} → Label: {series_colors[category_id]['label']}")
            
            # Second pass: create rows with colors
            for i, point in enumerate(lines_raw):
                if isinstance(point, dict):
                    bbox = point.get('bbox', [0, 0, 0, 0])
                    x_pixel = bbox[0] + bbox[2] / 2.0  # Center X
                    y_pixel = bbox[1] + bbox[3] / 2.0  # Center Y
                    category_id = point.get('category_id', 0)
                    
                    # Convert pixel Y to value using the calibrated function
                    point_value = y_to_val(y_pixel)
                    
                    # Get X-axis label for this point
                    x_category = line_x_labels.get(i, "")
                    
                    color_info = series_colors[category_id]
                    
                    rows.append({
                        "point_index": i,
                        "category": x_category,  # X-axis label (e.g., year)
                        "label": color_info['label'],  # Series name from legend
                        "color": color_info['hex'],
                        "x_pixel": round(x_pixel, 2),
                        "y_pixel": round(y_pixel, 2),
                        "value": None if point_value is None else round(point_value, 2),
                        "series_id": category_id,
                    })
        
        # Process Bar chart data
        for i, (x1, y1, x2, y2) in enumerate(bars_raw or []):
            rgb = bar_color(img, (x1, y1, x2, y2))
            match = closest_legend(rgb, legend_items)
            # y1 is the top of the bar (smaller Y value = higher in image = higher value)
            # y2 is the bottom of the bar (larger Y value = lower in image = lower value, usually baseline)
            val_top = y_to_val(y1)
            val_bottom = y_to_val(y2)
            
            # The bar value is what the top represents
            # Since bars typically start from baseline (~0) and grow upward,
            # the value at the top IS the bar's value
            bar_value = val_top
            
            # Get X-axis category label for this bar
            x_category = bar_x_labels.get(i, "")
            
            rows.append({
                "bar_index": i,
                "category": x_category,
                "label": match["text"] if match else "",
                "color": match["hex"] if match else rgb_to_hex(rgb),
                "x1": x1,
                "y1_top": y1,
                "x2": x2,
                "y2_bottom": y2,
                "pixel_height": (y2 - y1),
                "value": None if bar_value is None else round(bar_value, 2),
                "baseline_value": None if val_bottom is None else round(val_bottom, 2)
            })
        target_save = save_path
        csv_path = None
        if target_save is None:
            args_obj = globals().get("args")
            if args_obj is not None:
                target_save = getattr(args_obj, "save_path", None)
        df = pd.DataFrame(rows)
        if target_save:
            os.makedirs(target_save, exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            csv_path = os.path.join(target_save, f"{base}_table.csv")
            df.to_csv(csv_path, index=False)
            print(f"\nWrote: {csv_path}")
        
        # Print nice summary
        if len(rows) > 0:
            print('\n' + '=' * 100)
            if chart_type == "Pie":
                print(f'PIE CHART ANALYSIS: {os.path.basename(image_path)}')
                print('=' * 100)
                if titles:
                    print(f'Title: {list(titles.values())[0]}')
                print(f'Total segments detected: {len(df)}')
                if len(legend_items) > 0:
                    print(f'Legend items found: {len(legend_items)}')
                print()
                
                for idx, row in df.iterrows():
                    seg_num = idx + 1
                    angle = row.get('angle_degrees', 0)
                    percentage = row.get('value', 0)
                    label = row.get('label', '')
                    color = row.get('color', '')
                    
                    if label:
                        print(f'  Segment {seg_num:2d}: {label:15s} [{color}] = {percentage:6.2f}% ({angle:6.2f}°)')
                    else:
                        print(f'  Segment {seg_num:2d}: [{color}] = {percentage:6.2f}% ({angle:6.2f}°)')
                
                print('\n' + '-' * 100)
                print(f'  Total: {df["value"].sum():6.2f}%')
                print('=' * 100 + '\n')
            
            elif chart_type == "Line":
                # Sort by X position for chronological display
                df_sorted = df.sort_values('x_pixel').reset_index(drop=True)
                print(f'LINE CHART ANALYSIS: {os.path.basename(image_path)}')
                print('=' * 100)
                if titles:
                    print(f'Title: {list(titles.values())[0]}')
                print(f'Y-axis range: {y_min:,.0f} to {y_max:,.0f}' if y_min is not None and y_max is not None else 'Y-axis range: Not detected')
                print(f'Total data points detected: {len(df)}')
                if len(x_axis_labels) > 0:
                    print(f'X-axis categories detected: {len(x_axis_labels)} ({", ".join([xl["text"] for xl in x_axis_labels[:5]])}{"..." if len(x_axis_labels) > 5 else ""})')
                if len(legend_items) > 0:
                    print(f'Legend items found: {len(legend_items)}')
                
                # Group by series to show summary
                if 'series_id' in df.columns:
                    series_groups = df.groupby('label')
                    print(f'Number of series: {len(series_groups)}')
                print()
                
                # Display data organized by X-axis category, then by series
                if 'category' in df.columns and df['category'].notna().any():
                    # Group by X-axis category for better readability
                    categories = df_sorted['category'].unique()
                    
                    for cat in categories:
                        if cat and cat.strip():
                            print(f'\n  === X-axis: {cat} ===')
                            cat_data = df_sorted[df_sorted['category'] == cat]
                            
                            for idx, row in cat_data.iterrows():
                                label = row.get('label', '')
                                value = row.get('value')
                                color = row.get('color', '')
                                
                                val_str = f'{value:8,.2f}' if value is not None else '     N/A'
                                if label:
                                    print(f'    {label:15s} [{color}]: {val_str}')
                                else:
                                    print(f'    [{color}]: {val_str}')
                    
                    # Show any points without X-axis labels
                    no_cat = df_sorted[df_sorted['category'].isna() | (df_sorted['category'] == '')]
                    if len(no_cat) > 0:
                        print(f'\n  === Points without X-axis label ({len(no_cat)} points) ===')
                        current_series = None
                        for idx, row in no_cat.iterrows():
                            label = row.get('label', '')
                            value = row.get('value')
                            x_pixel = row.get('x_pixel', 0)
                            color = row.get('color', '')
                            
                            if label != current_series:
                                if current_series is not None:
                                    print()
                                print(f'    --- {label} [{color}] ---')
                                current_series = label
                            
                            val_str = f'{value:8,.2f}' if value is not None else '     N/A'
                            print(f'    X={x_pixel:7.1f} → Y = {val_str}')
                else:
                    # Fallback: Display by series
                    current_series = None
                    for idx, row in df_sorted.iterrows():
                        point_num = idx + 1
                        label = row.get('label', '')
                        value = row.get('value')
                        x_pixel = row.get('x_pixel', 0)
                        color = row.get('color', '')
                        
                        # Print series header when switching to new series
                        if label != current_series:
                            if current_series is not None:
                                print()
                            print(f'  --- {label} [{color}] ---')
                            current_series = label
                        
                        val_str = f'{value:8,.2f}' if value is not None else '     N/A'
                        print(f'  Point {point_num:3d}: X={x_pixel:7.1f} → Y = {val_str}')
                
                print('\n' + '-' * 100)
                # Filter out None values for statistics
                valid_values = df['value'].dropna()
                if len(valid_values) > 0:
                    print(f'  Min value: {valid_values.min():10,.2f}')
                    print(f'  Max value: {valid_values.max():10,.2f}')
                    print(f'  Average:   {valid_values.mean():10,.2f}')
                else:
                    print('  No valid Y-values detected (Y-axis calibration may have failed)')
                
                # Show per-series statistics if multiple series
                if 'series_id' in df.columns and df['series_id'].nunique() > 1:
                    print('\n  Per-Series Statistics:')
                    for series_label in df['label'].unique():
                        series_data = df[df['label'] == series_label]['value'].dropna()
                        if len(series_data) > 0:
                            print(f'    {series_label:15s}: Min={series_data.min():8,.2f}, Max={series_data.max():8,.2f}, Avg={series_data.mean():8,.2f}')
                
                print('=' * 100 + '\n')
            
            else:  # Bar chart
                df_sorted = df.sort_values('x1').reset_index(drop=True)
                print(f'BAR CHART ANALYSIS: {os.path.basename(image_path)}')
                print('=' * 100)
                if titles:
                    print(f'Title: {list(titles.values())[0]}')
                print(f'Y-axis range: {y_min:,.0f} to {y_max:,.0f}' if y_min is not None and y_max is not None else 'Y-axis range: Not detected')
                print(f'Total bars detected: {len(df)}')
                if len(x_axis_labels) > 0:
                    print(f'X-axis categories detected: {len(x_axis_labels)}')
                if len(legend_items) > 0:
                    print(f'Legend items found: {len(legend_items)}')
                print()
                
                for idx, row in df_sorted.iterrows():
                    bar_num = idx + 1
                    category = row['category'] if row['category'] else '(no X-label)'
                    label = row['label'] if row['label'] else ''
                    value = row['value']
                    x_center = (row['x1'] + row['x2']) / 2
                    color = row.get('color', '')
                    
                    # Format output based on what labels are available
                    if category and label:
                        print(f'  Bar {bar_num:2d}: {category:15s} | {label:15s} [{color}] = {value:8,.2f}')
                    elif category:
                        print(f'  Bar {bar_num:2d}: {category:15s} [{color}] = {value:8,.2f}')
                    elif label:
                        print(f'  Bar {bar_num:2d}: {label:15s} [{color}] = {value:8,.2f}')
                    else:
                        print(f'  Bar {bar_num:2d}: [{color}] = {value:8,.2f}')
                
                print('\n' + '-' * 100)
                print(f'  Min value: {df["value"].min():10,.2f}')
                print(f'  Max value: {df["value"].max():10,.2f}')
                print(f'  Average:   {df["value"].mean():10,.2f}')
                print('=' * 100 + '\n')

    result = {
        "chart_type": chart_type,
        "chart_title_candidates": titles,
        "y_axis_min_est": y_min,
        "y_axis_max_est": y_max,
    }
    
    # Add chart-specific data
    if chart_type == "Bar":
        result["bars_raw"] = _to_py(bars_raw)
        result["bars_summary"] = rows
    elif chart_type == "Line":
        result["lines_raw"] = _to_py(lines_raw)
        result["lines_summary"] = rows  # List with extracted point values
    elif chart_type == "Pie":
        result["pie_segments"] = _to_py(pie_segments)  # List of angles
        result["pie_summary"] = rows  # List with percentages calculated from angles
    
    if csv_path:
        result["csv_path"] = csv_path
    if return_images:
        result["original_image_b64"] = original_b64
        result["overlay_image_b64"] = overlay_b64
    return result


# ------------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DeepRule ChartOCR (Bar, Line, Pie)")
    parser.add_argument("--image_path",  default="test",
                        help="Folder containing chart images to process")
    parser.add_argument("--save_path",   default="save",
                        help="Folder where JSON/CSVs will be written")
    parser.add_argument("--type",        default="Bar", 
                        choices=["Bar", "Line", "Pie"],
                        help="Chart type: Bar, Line, or Pie")
    parser.add_argument("--data_dir",    default=".",
                        help="Root dir that matches model configs (e.g. /content/DeepRule-master)")
    parser.add_argument("--cache_path",  default="./cache",
                        help="Where pretrained weights/cache live (e.g. /content/DeepRule-master/cache)")
    parser.add_argument("--result_path", default=None, type=str,
                        help="(Optional) if set, also write a single JSON here")
    return parser.parse_args()


if __name__=="__main__":
    os.environ.setdefault("MPLBACKEND","Agg")
    os.environ.setdefault("TESSDATA_PREFIX","/usr/share/tesseract-ocr/4.00/tessdata")
    args=parse_args()

    methods=Pre_load_nets(args.type,0,args.data_dir,args.cache_path)
    make_dirs([args.save_path])

    all_results={}
    for f in tqdm(sorted(os.listdir(args.image_path))):
        ip=os.path.join(args.image_path,f)
        if not os.path.isfile(ip): continue
        try: res=run_on_image(ip,args.type,save_path=args.save_path)
        except Exception as e: res={"error":str(e)}
        all_results[f]=res
        with open(os.path.join(args.save_path,os.path.splitext(f)[0]+"_result.json"),"w") as jf:
            json.dump(_to_py(res),jf,indent=2)
    with open(os.path.join(args.save_path,"all_results.json"),"w") as f:
        json.dump(_to_py(all_results),f,indent=2)
