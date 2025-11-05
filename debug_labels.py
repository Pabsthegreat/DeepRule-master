#!/usr/bin/env python
"""Debug Y-axis and bar label detection"""
import cv2
import pytesseract
from PIL import Image
import re

img_path = 'sample_images/bar/8a65d61e46f8f92a4c5be2ed012f7d8c_d3d3LnN0YXRpc3RpcXVlcy5kZXZlbG9wcGVtZW50LWR1cmFibGUuZ291di5mcgkzNy4yMzUuODkuMTA3.xls-1-0.png'

# Load image
img = cv2.imread(img_path)
pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Run OCR on full image
data = pytesseract.image_to_data(pil_img, lang="eng", output_type=pytesseract.Output.DICT)

print("All text with percentages or numbers:")
print("=" * 80)
for i, txt in enumerate(data["text"]):
    txt = (txt or "").strip()
    if not txt:
        continue
    if '%' in txt or re.search(r'\d+[,.]?\d*', txt):
        x, y = data["left"][i], data["top"][i]
        print(f'"{txt}" at (x={x}, y={y})')

print("\n\nX-axis area (y > 450):")
print("=" * 80)
for i, txt in enumerate(data["text"]):
    txt = (txt or "").strip()
    if not txt or len(txt) < 2:
        continue
    y = data["top"][i]
    x = data["left"][i]
    if y > 450:
        print(f'"{txt}" at (x={x}, y={y})')
