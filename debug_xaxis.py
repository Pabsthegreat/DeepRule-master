#!/usr/bin/env python3
"""
Debug script to test X-axis tick extraction with detailed visualization
"""

import sys
import os
import cv2
import torch
from PIL import Image
from test_pipe_type_cloud import ocr_xaxis_region_with_rotation, Pre_load_nets, GroupCls, _to_box

def main():
    if len(sys.argv) < 2:
        print("Usage: python debug_xaxis.py <image_path>")
        print("\nExample:")
        print("  python debug_xaxis.py sample_images/line/f4a521cd7baa40c6422d0f3c55adbf7c_d3d3LndhdGVyLmNhLmdvdgkxMzYuMjAwLjI0My4yNTQ=.xls-1-3.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Error: Cannot read image: {image_path}")
        sys.exit(1)
    
    H, W = img.shape[:2]
    print(f"📊 Image loaded: {W}x{H} pixels")
    print(f"📁 Path: {image_path}")
    
    # Load the chart detection model to get REAL plot coordinates
    print(f"\n🔄 Loading chart detection model...")
    model_bundle = Pre_load_nets(
        chart_type="Line",
        id_cuda=0,
        data_dir=".",
        cache_dir="./cache/nnet"
    )
    
    # Detect plot region using the model
    print(f"🔍 Detecting plot region...")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    with torch.no_grad():
        cls_db, cls_net, cls_fn = model_bundle["Cls"]
        cls_res = cls_fn(img, cls_db, cls_net, debug=False)
        tls, brs = cls_res[1], cls_res[2]
        _, raw_info = GroupCls(pil_img, tls, brs)
        cls_info = {int(k): _to_box(v) for k,v in raw_info.items()}
    
    # Get the actual plot region from the model
    if 5 not in cls_info:
        print(f"❌ Error: Could not detect plot region (cls_info[5] not found)")
        print(f"   Available regions: {list(cls_info.keys())}")
        sys.exit(1)
    
    plot_bbox = _to_box(cls_info[5])
    if plot_bbox is None:
        print(f"❌ Error: Plot region is None")
        sys.exit(1)
    
    print(f"\n🎯 Using ACTUAL plot region from model detection:")
    print(f"   x: {plot_bbox[0]:.0f} -> {plot_bbox[2]:.0f}")
    print(f"   y: {plot_bbox[1]:.0f} -> {plot_bbox[3]:.0f}")
    print(f"\n{'='*80}")
    print("STARTING X-AXIS TICK EXTRACTION WITH DEBUG")
    print(f"{'='*80}")
    
    # Run the X-axis OCR with debug enabled
    results = ocr_xaxis_region_with_rotation(image_path, plot_bbox, debug=True)
    
    print(f"\n✅ Extraction complete!")
    print(f"\n📋 Summary: {len(results)} X-axis labels detected")
    
    if results:
        print("\n🏷️  Detected labels:")
        for i, result in enumerate(results):
            print(f"   [{i}] {result['text']}")
    else:
        print("\n⚠️  No X-axis labels detected")
    
    print(f"\n💾 Debug images saved to debug_output/xaxis_*.png")
    print("   You can view them with: open debug_output/xaxis_*.png")
    print(f"   Or in VS Code: {os.path.abspath('debug_output')}")

if __name__ == "__main__":
    main()
