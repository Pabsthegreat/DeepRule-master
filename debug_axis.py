#!/usr/bin/env python
"""Debug script to analyze Y-axis values"""
import cv2
import numpy as np
from PIL import Image
import pytesseract
import json

image_path = "sample_images/bar/8a2b5419565a27b03f57c4ab22f2c78a_d3d3LnByb2plY3RzYnlqZW4uY29tCTE4MS4yMjQuMTM2LjQ3.xls-0-0.png"

# Load the image
img = cv2.imread(image_path)
H, W = img.shape[:2]

print(f"Image size: {W} x {H}")

# Load the results to get plot region
with open('output_bar_improved/all_results.json', 'r') as f:
    results = json.load(f)
    filename = list(results.keys())[0]
    result = results[filename]
    print(f"\nY-axis estimates: {result['y_axis_min_est']} to {result['y_axis_max_est']}")

# Try OCR on left side of image
left_strip = img[:, 0:100]
cv2.imwrite('/tmp/left_strip.png', left_strip)
print(f"\nSaved left strip to /tmp/left_strip.png")

# Run OCR on left strip
pil_left = Image.fromarray(cv2.cvtColor(left_strip, cv2.COLOR_BGR2RGB))
data = pytesseract.image_to_data(pil_left, output_type=pytesseract.Output.DICT)

print("\nAll OCR results from left strip:")
for i, txt in enumerate(data["text"]):
    if txt.strip():
        print(f"  [{i}] '{txt}' at y={data['top'][i]}, conf={data['conf'][i]}")

# Try with upscaling
print("\nWith 2x upscaling:")
left_strip_2x = cv2.resize(left_strip, (0,0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
pil_left_2x = Image.fromarray(cv2.cvtColor(left_strip_2x, cv2.COLOR_BGR2RGB))
data2 = pytesseract.image_to_data(pil_left_2x, output_type=pytesseract.Output.DICT)

for i, txt in enumerate(data2["text"]):
    if txt.strip():
        y_original = data2['top'][i] / 2.0
        print(f"  [{i}] '{txt}' at y={y_original:.1f} (original scale), conf={data2['conf'][i]}")

# Try OCR on full image to see all text
print("\n\nFull image OCR (all text):")
pil_full = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
data_full = pytesseract.image_to_data(pil_full, output_type=pytesseract.Output.DICT)

for i, txt in enumerate(data_full["text"]):
    txt_clean = txt.strip()
    if txt_clean and data_full['left'][i] < 80:  # Focus on left side
        print(f"  '{txt_clean}' at x={data_full['left'][i]}, y={data_full['top'][i]}")
