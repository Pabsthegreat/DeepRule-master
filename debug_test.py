#!/usr/bin/env python
"""Debug script to test PIL image handling"""
import os
import sys
from PIL import Image
import cv2
import numpy as np

image_path = "/Users/adarsh/Documents/GitHub/DeepRule-master/sample_images/bar/8a2b5419565a27b03f57c4ab22f2c78a_d3d3LnByb2plY3RzYnlqZW4uY29tCTE4MS4yMjQuMTM2LjQ3.xls-0-0.png"

print("Testing PIL Image loading...")
pil_img = Image.open(image_path)
print(f"PIL Image type: {type(pil_img)}")
print(f"PIL Image mode: {pil_img.mode}")
print(f"PIL Image size: {pil_img.size}")
print(f"Has load attr: {hasattr(pil_img, 'load')}")

# Try loading pixel data
if hasattr(pil_img, 'load'):
    pil_img.load()
    print("PIL image loaded successfully")

print("\nTesting cv2 imread...")
img = cv2.imread(image_path)
print(f"cv2 image type: {type(img)}")
print(f"cv2 image shape: {img.shape if img is not None else 'None'}")

# Test GroupCls imports
print("\nTesting GroupCls import...")
try:
    from RuleGroup.Cls import GroupCls
    print("GroupCls imported successfully")
    
    # Create dummy tls/brs dictionaries
    tls_dummy = {1: np.array([[0.5, 1.0, 100.0, 100.0]])}
    brs_dummy = {1: np.array([[0.5, 1.0, 200.0, 200.0]])}
    
    print("\nTesting GroupCls call...")
    result_img, raw_info = GroupCls(pil_img, tls_dummy, brs_dummy)
    print(f"GroupCls returned: image type={type(result_img)}, info={raw_info}")
except Exception as e:
    print(f"Error in GroupCls: {e}")
    import traceback
    traceback.print_exc()
