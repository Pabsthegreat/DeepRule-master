#!/usr/bin/env python
import os
import re
import cv2
import json
import math
import torch
import numpy as np
import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import traceback

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


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------

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

def Pre_load_nets(chart_type, id_cuda, data_dir, cache_dir):
    methods = {}
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", data_dir, cache_dir, id_cuda)
    path_cls = "testfile.test_CornerNetCls"
    testing_cls = importlib.import_module(path_cls).testing
    methods["Cls"] = [db_cls, nnet_cls, testing_cls]

    if chart_type == "Bar":
        db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", data_dir, cache_dir, id_cuda)
        path_bar = "testfile.test_CornerNetPureBar"
        testing_bar = importlib.import_module(path_bar).testing
        methods["Bar"] = [db_bar, nnet_bar, testing_bar]
    return methods


# ------------------------------------------------------------------
# OCR helpers
# ------------------------------------------------------------------

def ocr_result_full_image(image_path):
    print(f"  [DEBUG] Starting OCR on {image_path}")
    os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Cannot read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    print(f"  [DEBUG] PIL image type for OCR: {type(pil)}")
    data = pytesseract.image_to_data(pil, lang="eng", output_type=pytesseract.Output.DICT)
    words = []
    for i, txt in enumerate(data["text"]):
        txt = (txt or "").strip()
        if not txt:
            continue
        x, y, w, h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({"text": txt, "bbox": [x, y, x + w, y + h]})
    print(f"  [DEBUG] OCR found {len(words)} words")
    return words

def extract_titles_from_clsinfo(cls_info, word_infos):
    print(f"  [DEBUG] Extracting titles from cls_info keys: {list(cls_info.keys())}")
    out = {}
    for rid in [1, 2, 3]:
        if rid not in cls_info:
            continue
        rbox = _to_box(cls_info[rid])
        captured = [(w["text"], w["bbox"][1], w["bbox"][0]) for w in word_infos if boxes_intersection_ratio(rbox, w["bbox"]) > 0.5]
        captured.sort(key=lambda x: (x[1], x[2]))
        if captured:
            out[str(rid)] = " ".join([c[0] for c in captured])
    print(f"  [DEBUG] Extracted titles: {out}")
    return out

def extract_axis_scale_from_clsinfo(image, cls_info):
    print(f"  [DEBUG] Extracting axis scale, cls_info keys: {list(cls_info.keys())}")
    if 5 not in cls_info:
        print(f"  [DEBUG] No plot region (5) found")
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
    if not nums: 
        print(f"  [DEBUG] No axis numbers found")
        return None, None
    print(f"  [DEBUG] Axis scale: min={min(nums)}, max={max(nums)}")
    return min(nums), max(nums)


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
    print(f"  [DEBUG] Found {len(legend)} legend items")
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

def run_on_image(image_path, chart_type="Bar"):
    print(f"\n[DEBUG] Processing: {image_path}")
    pil_img = Image.open(image_path)
    print(f"[DEBUG] PIL image opened: type={type(pil_img)}, mode={pil_img.mode}, size={pil_img.size}")
    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"Could not read {image_path}")
    print(f"[DEBUG] cv2 image loaded: shape={img.shape}")

    with torch.no_grad():
        print(f"[DEBUG] Running Cls model...")
        cls_db, cls_net, cls_fn = methods["Cls"]
        cls_res = cls_fn(img, cls_db, cls_net, debug=False)
        print(f"[DEBUG] Cls result type: {type(cls_res)}, len: {len(cls_res) if isinstance(cls_res, (list, tuple)) else 'N/A'}")
        tls, brs = cls_res[1], cls_res[2]
        print(f"[DEBUG] tls type: {type(tls)}, brs type: {type(brs)}")
        
        print(f"[DEBUG] Calling GroupCls...")
        _, raw_info = GroupCls(pil_img, tls, brs)
        print(f"[DEBUG] GroupCls returned raw_info: {raw_info}")
        cls_info = {int(k): _to_box(v) for k,v in raw_info.items()}
        print(f"[DEBUG] cls_info after boxing: {cls_info}")

        print(f"[DEBUG] Running OCR...")
        words = ocr_result_full_image(image_path)
        
        print(f"[DEBUG] Extracting titles...")
        titles = extract_titles_from_clsinfo(cls_info, words)
        
        print(f"[DEBUG] Extracting axis scale...")
        y_min, y_max = extract_axis_scale_from_clsinfo(img, cls_info)

        bars_raw = []
        if chart_type == "Bar":
            print(f"[DEBUG] Running Bar model...")
            bdb, bnet, bfn = methods["Bar"]
            bres = bfn(img, bdb, bnet, debug=False)
            tls_b, brs_b = bres[0], bres[1]
            print(f"[DEBUG] Bar result - tls_b type: {type(tls_b)}, brs_b type: {type(brs_b)}")
            bars_raw = GroupBarRaw(img, tls_b, brs_b)
            print(f"[DEBUG] GroupBarRaw returned {len(bars_raw) if bars_raw else 0} bars")

        print(f"[DEBUG] Finding legend pairs...")
        legend_items = find_legend_pairs(img, words)

        # Fit pixel→value regression
        tick_pairs = []
        if 5 in cls_info:
            x1,y1,x2,y2 = cls_info[5]
            strip = img[int(y1):int(y2), max(0,int(x1)-70):int(x1)]
            data = pytesseract.image_to_data(
                Image.fromarray(cv2.cvtColor(cv2.resize(strip,(0,0),fx=2,fy=2), cv2.COLOR_BGR2RGB)),
                config="--psm 6 -c tessedit_char_whitelist=0123456789.-",
                output_type=pytesseract.Output.DICT)
            for i, txt in enumerate(data["text"]):
                txt = (txt or "").strip()
                if not txt: continue
                try: val=float(re.sub(r"[^0-9.\-]", "", txt))
                except: continue
                oy=int(y1)+int(data["top"][i]/2)+int(data["height"][i]/4)
                tick_pairs.append((oy,val))
        print(f"[DEBUG] Found {len(tick_pairs)} tick pairs")
        
        a=b=None
        if len(tick_pairs)>=2:
            ys=np.array([p[0] for p in tick_pairs])
            vs=np.array([p[1] for p in tick_pairs])
            A=np.vstack([ys,np.ones_like(ys)]).T
            a,b=np.linalg.lstsq(A,vs,rcond=None)[0]
            print(f"[DEBUG] Linear regression: a={a}, b={b}")
        def y_to_val(y): return float(a*y+b) if a is not None else None

        rows=[]
        for i,(x1,y1,x2,y2) in enumerate(bars_raw or []):
            rgb=bar_color(img,(x1,y1,x2,y2))
            match=closest_legend(rgb,legend_items)
            val=y_to_val(y1)
            rows.append({
                "bar_index":i,
                "label":match["text"] if match else "",
                "color":match["hex"] if match else rgb_to_hex(rgb),
                "x1":x1,"y1_top":y1,"x2":x2,"y2_bottom":y2,
                "pixel_height":(y2-y1),
                "value_estimate":None if val is None else round(val,2)
            })
        print(f"[DEBUG] Generated {len(rows)} data rows")
        
        base=os.path.splitext(os.path.basename(image_path))[0]
        csv_out=os.path.join(args.save_path,f"{base}_table.csv")
        pd.DataFrame(rows).to_csv(csv_out,index=False)
        print(f"[DEBUG] Wrote CSV: {csv_out}")

        return {
            "chart_type":chart_type,
            "chart_title_candidates":titles,
            "y_axis_min_est":y_min,
            "y_axis_max_est":y_max,
            "bars_raw":_to_py(bars_raw)
        }


# ------------------------------------------------------------------
# CLI / main
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DeepRule ChartOCR (Bar only) - DEBUG VERSION")
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

    print("[DEBUG] Loading models...")
    methods=Pre_load_nets(args.type,0,args.data_dir,args.cache_path)
    make_dirs([args.save_path])

    all_results={}
    files_to_process = sorted(os.listdir(args.image_path))
    print(f"[DEBUG] Found {len(files_to_process)} items in {args.image_path}")
    
    for f in tqdm(files_to_process):
        ip=os.path.join(args.image_path,f)
        if not os.path.isfile(ip): 
            print(f"[DEBUG] Skipping {f} (not a file)")
            continue
        try: 
            res=run_on_image(ip,args.type)
            print(f"[DEBUG] SUCCESS for {f}")
        except Exception as e: 
            print(f"[DEBUG] ERROR for {f}: {e}")
            traceback.print_exc()
            res={"error":str(e)}
        all_results[f]=res
        with open(os.path.join(args.save_path,os.path.splitext(f)[0]+"_result.json"),"w") as jf:
            json.dump(_to_py(res),jf,indent=2)
    
    print(f"\n[DEBUG] Processed {len(all_results)} files")
    with open(os.path.join(args.save_path,"all_results.json"),"w") as f:
        json.dump(_to_py(all_results),f,indent=2)
    print(f"[DEBUG] Wrote all_results.json")
