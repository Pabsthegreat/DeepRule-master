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

    _methods_cache[cache_key] = methods_local
    methods = methods_local
    return methods


# ------------------------------------------------------------------
# OCR helpers
# ------------------------------------------------------------------

def ocr_result_full_image(image_path):
    os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    data = pytesseract.image_to_data(pil, lang="eng", output_type=pytesseract.Output.DICT)
    words = []
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

def extract_xaxis_labels(word_infos, cls_info, img_height):
    """Extract X-axis labels (category labels below the plot area)"""
    if 5 not in cls_info:
        return []
    
    plot = _to_box(cls_info[5])
    if plot is None:
        return []
    
    x1_plot, y1_plot, x2_plot, y2_plot = plot
    
    # Look for text below the plot area
    candidate_words = []
    for w in word_infos:
        x_center = (w["bbox"][0] + w["bbox"][2]) / 2
        y_center = (w["bbox"][1] + w["bbox"][3]) / 2
        
        # Check if text is below plot area and horizontally within it
        text = w["text"].strip()
        
        # Exclude only percentage signs with decimals (like "14.5%")
        # Allow years (4-digit numbers) and other integers as they are likely X-axis labels
        is_percentage = '%' in text and '.' in text  # Only exclude decimal percentages
        
        if (y_center > y2_plot and 
            x_center >= x1_plot and 
            x_center <= x2_plot and
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
                "bbox": w["bbox"]
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
    
    # Group words that are horizontally close together
    # Use adaptive gap detection: small gaps = same label, large gaps = different labels
    grouped_labels = []
    i = 0
    while i < len(main_row_words):
        current = main_row_words[i]
        group = [current]
        j = i + 1
        
        # Look for nearby words on the same line
        while j < len(main_row_words):
            next_word = main_row_words[j]
            # Check if horizontally close
            x_gap = next_word["x_left"] - current["x_right"]
            
            # Small gap means same label, but break on large gaps
            # Typical word spacing is < 10px, label separation is > 30px
            if x_gap < 15 and x_gap > -5:  # Very close = same label
                group.append(next_word)
                current = next_word
                j += 1
            elif x_gap >= 15:  # Large gap = new label
                break
            else:
                j += 1
        
        # Combine grouped words
        if group:
            combined_text = " ".join([w["text"] for w in group])
            avg_x = sum([w["x_center"] for w in group]) / len(group)
            grouped_labels.append({
                "text": combined_text,
                "x_center": avg_x,
                "y_pos": group[0]["y_pos"],
                "bbox": [group[0]["bbox"][0], group[0]["bbox"][1], 
                        group[-1]["bbox"][2], group[-1]["bbox"][3]]
            })
        
        i = j if j > i + 1 else i + 1
    
    # Sort by X position (left to right)
    grouped_labels.sort(key=lambda x: x["x_center"])
    return grouped_labels

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


# ------------------------------------------------------------------
# Legend / color matching
# ------------------------------------------------------------------

def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*[int(v) for v in rgb])

def bar_color(img, box):
    x1,y1,x2,y2 = [int(v) for v in box]
    roi = img[y1:y2, x1:x2]
    if roi.size == 0: return (0,0,0)
    h,w = roi.shape[:2]
    ys, ye = int(h*0.2), int(h*0.8)
    xs, xe = int(w*0.2), int(w*0.8)
    core = roi[ys:ye, xs:xe]
    bgr = core.reshape(-1,3).mean(axis=0)
    return (bgr[2], bgr[1], bgr[0])

def find_legend_pairs(img, words):
    H, W = img.shape[:2]
    legend = []
    for w in words:
        x1,y1,x2,y2 = w["bbox"]
        if x1 >= int(W*0.65) and len(w["text"]) >= 2:
            cx = max(0, x1 - 12)
            cy = int((y1+y2)/2)
            sx1, sy1 = max(0, cx-6), max(0, cy-6)
            sx2, sy2 = min(W-1, cx+6), min(H-1, cy+6)
            patch = img[sy1:sy2, sx1:sx2]
            if patch.size == 0: continue
            bgr = patch.reshape(-1,3).mean(axis=0)
            rgb = (bgr[2], bgr[1], bgr[0])
            legend.append({"text": w["text"], "rgb": rgb, "hex": rgb_to_hex(rgb)})
    return legend

def closest_legend(rgb, legend):
    if not legend: return None
    bx,by,bz = rgb
    best, best_d = None, 1e9
    for item in legend:
        rx,ry,rz = item["rgb"]
        d = (bx-rx)**2+(by-ry)**2+(bz-rz)**2
        if d<best_d:
            best,best_d=item,d
    return best


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
        if chart_type == "Bar" and "Bar" in model_bundle:
            bdb, bnet, bfn = model_bundle["Bar"]
            bres = bfn(img, bdb, bnet, debug=False)
            tls_b, brs_b = bres[0], bres[1]
            # Convert cv2 image to PIL for GroupBarRaw (it expects PIL Image)
            pil_img_copy = pil_img.copy() if hasattr(pil_img, 'copy') else Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            bars_raw = GroupBarRaw(pil_img_copy, tls_b, brs_b)
            if return_images:
                overlay_b64 = _encode_image_to_base64(pil_img_copy)

        # Extract X-axis labels (category labels)
        x_axis_labels = extract_xaxis_labels(words, cls_info, img.shape[0])
        bar_x_labels = map_labels_to_bars(bars_raw, x_axis_labels)
        
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

        rows = []
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
            df_sorted = df.sort_values('x1').reset_index(drop=True)
            print('\n' + '=' * 100)
            print(f'BAR CHART ANALYSIS: {os.path.basename(image_path)}')
            print('=' * 100)
            if titles:
                print(f'Title: {list(titles.values())[0]}')
            print(f'Y-axis range: {y_min:,.0f} to {y_max:,.0f}' if y_min is not None and y_max is not None else 'Y-axis range: Not detected')
            print(f'Total bars detected: {len(df)}')
            if len(x_axis_labels) > 0:
                print(f'X-axis categories detected: {len(x_axis_labels)}')
            print()
            
            for idx, row in df_sorted.iterrows():
                bar_num = idx + 1
                category = row['category'] if row['category'] else '(no X-label)'
                label = row['label'] if row['label'] else ''
                value = row['value']
                x_center = (row['x1'] + row['x2']) / 2
                
                # Format output based on what labels are available
                if category and label:
                    print(f'  Bar {bar_num:2d}: {category:15s} | {label:12s} = {value:8,.2f}')
                elif category:
                    print(f'  Bar {bar_num:2d}: {category:15s} = {value:8,.2f}')
                elif label:
                    print(f'  Bar {bar_num:2d} at X={x_center:6.1f}: {label:12s} = {value:8,.2f}')
                else:
                    print(f'  Bar {bar_num:2d} at X={x_center:6.1f}: {value:8,.2f}')
            
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
        "bars_raw": _to_py(bars_raw),
        "bars_summary": rows,
    }
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
    parser = argparse.ArgumentParser(description="DeepRule ChartOCR (Bar only)")
    parser.add_argument("--image_path",  default="test",
                        help="Folder containing chart images to process")
    parser.add_argument("--save_path",   default="save",
                        help="Folder where JSON/CSVs will be written")
    parser.add_argument("--type",        default="Bar",
                        help="Chart type hint (currently only 'Bar' supported)")
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
