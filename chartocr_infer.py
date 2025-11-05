# chartocr_infer.py
import os, json, re, math
import numpy as np
import cv2
import torch
from PIL import Image
import pytesseract
import importlib

from config import system_configs
from nnet.py_factory import NetworkFactory
from db.datasets import datasets

from RuleGroup.Cls import GroupCls
from RuleGroup.Bar import GroupBarRaw

torch.backends.cudnn.benchmark = False


# ----------------------------- small helpers -----------------------------
def _to_py(o):
    """Recursively convert numpy types → native Python for json.dump."""
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
    """
    Normalize coords into [x1,y1,x2,y2].
    Accepts 4-tuple boxes or 8-tuple polygons.
    """
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
    """
    Return intersection_area / area_of_small for [x1,y1,x2,y2] boxes.
    """
    B = _to_box(box_big)
    S = _to_box(box_small)
    if B is None or S is None:
        return 0.0
    bx1, by1, bx2, by2 = B
    sx1, sy1, sx2, sy2 = S
    ix1 = max(bx1, sx1)
    iy1 = max(by1, sy1)
    ix2 = min(bx2, sx2)
    iy2 = min(by2, sy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    small = max(1.0, (sx2 - sx1) * (sy2 - sy1))
    return inter / small


# ----------------------------- model loading -----------------------------
def load_net(testiter, cfg_name, data_dir, cache_dir, cuda_id=0):
    """
    Load a CornerNet-style model + weights, DeepRule style.
    """
    cfg_file = os.path.join(system_configs.config_dir, cfg_name + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)

    # override runtime paths
    configs["system"]["snapshot_name"] = cfg_name
    configs["system"]["data_dir"] = data_dir
    configs["system"]["cache_dir"] = cache_dir
    configs["system"]["result_dir"] = "result_dir"
    configs["system"]["tar_data_dir"] = "Cls"
    system_configs.update_config(configs["system"])

    split = {
        "training": system_configs.train_split,
        "validation": system_configs.val_split,
        "testing": system_configs.test_split,
    }["validation"]

    test_iter = system_configs.max_iter if testiter is None else testiter
    print(f"[ChartOCR] loading params at iter {test_iter} for {cfg_name}")

    dataset = system_configs.dataset
    db = datasets[dataset](configs["db"], split)

    nnet = NetworkFactory(db)
    nnet.load_params(test_iter)
    if torch.cuda.is_available() and cuda_id is not None:
        nnet.cuda(cuda_id)
    nnet.eval_mode()
    return db, nnet


def preload_methods(chart_type, cuda_id, data_dir, cache_dir):
    """
    Build the `methods` dict used by inference.
    - Always loads 'Cls'
    - Loads 'Bar' when chart_type == 'Bar'
    """
    methods = {}

    # Cls
    db_cls, nnet_cls = load_net(50000, "CornerNetCls", data_dir, cache_dir, cuda_id)
    t_cls = importlib.import_module("testfile.test_CornerNetCls").testing
    methods["Cls"] = [db_cls, nnet_cls, t_cls]

    # Bar
    if chart_type == "Bar":
        db_bar, nnet_bar = load_net(50000, "CornerNetPureBar", data_dir, cache_dir, cuda_id)
        t_bar = importlib.import_module("testfile.test_CornerNetPureBar").testing
        methods["Bar"] = [db_bar, nnet_bar, t_bar]

    return methods


# ----------------------------- OCR helpers -----------------------------
def ocr_words(image_path):
    """
    OCR the full image → list of {text, bbox=[x1,y1,x2,y2]}.
    """
    # Ensure tessdata. On macOS (brew), tessdata is typically set already; keeping a default fallback.
    os.environ.setdefault("TESSDATA_PREFIX", "/usr/share/tesseract-ocr/4.00/tessdata")

    img = cv2.imread(image_path)
    if img is None:
        raise RuntimeError(f"cannot read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    data = pytesseract.image_to_data(
        Image.fromarray(rgb),
        lang="eng",
        output_type=pytesseract.Output.DICT
    )

    words = []
    n = len(data["text"])
    for i in range(n):
        txt = (data["text"][i] or "").strip()
        if not txt:
            continue
        x = int(data["left"][i])
        y = int(data["top"][i])
        w = int(data["width"][i])
        h = int(data["height"][i])
        words.append({"text": txt, "bbox": [x, y, x + w, y + h]})
    return words


def extract_titles(cls_info, word_infos):
    """
    Build strings for regions 1,2,3 by gathering OCR words inside those boxes.
    """
    out = {}
    for rid in [1, 2, 3]:
        if rid not in cls_info:
            continue
        rbox = _to_box(cls_info[rid])
        if rbox is None:
            continue

        captured = []
        for w in word_infos:
            if boxes_intersection_ratio(rbox, w["bbox"]) > 0.5:
                captured.append((w["text"], w["bbox"][1], w["bbox"][0]))
        captured.sort(key=lambda x: (x[1], x[2]))
        if captured:
            out[str(rid)] = " ".join(c[0] for c in captured)
    return out


def extract_y_axis_range(img_bgr, cls_info):
    """
    Estimate y-axis [min, max] from text along the left of the plot area (id=5).
    """
    if 5 not in cls_info:
        return (None, None)

    plot = _to_box(cls_info[5])
    if plot is None:
        return (None, None)
    x1, y1, x2, y2 = map(int, plot)

    # thin strip left of the plot
    sx1, sx2 = max(x1 - 60, 0), x1
    crop = img_bgr[y1:y2, sx1:sx2]
    if crop.size == 0:
        return (None, None)

    up = cv2.resize(crop, (0, 0), fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    up_rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        Image.fromarray(up_rgb),
        lang="eng",
        config="--psm 6",
        output_type=pytesseract.Output.DICT
    )

    vals = []
    for t in data["text"]:
        if not t:
            continue
        c = re.sub(r"[^0-9.\-]", "", t)
        if re.search(r"[0-9]", c):
            try:
                vals.append(float(c))
            except:
                pass

    if not vals:
        return (None, None)

    lo, hi = min(vals), max(vals)
    if abs(hi - lo) < 1e-6:
        return (None, None)
    return lo, hi


# ----------------------------- main per-image -----------------------------
def run_on_image(image_path, methods, chart_type="Bar"):
    """
    Full pipeline for one image:
      - Cls net (layout) → GroupCls → cls_info
      - OCR → titles
      - y-axis rough min/max
      - Bar net → GroupBarRaw
    Returns a JSON-serializable dict.
    """
    pil = Image.open(image_path)
    im = cv2.imread(image_path)
    if im is None:
        raise RuntimeError(f"Could not read image at {image_path}")

    with torch.no_grad():
        # Cls / layout
        cls_db, cls_net, cls_test = methods["Cls"]
        cls_results = cls_test(im, cls_db, cls_net, debug=False)   # original returns [info, tls, brs]
        tls, brs = cls_results[1], cls_results[2]

        _, cls_info0 = GroupCls(pil, tls, brs)  # might contain polys; normalize
        cls_info = {int(k): _to_box(v) for k, v in cls_info0.items()}

        # OCR titles/labels
        words = ocr_words(image_path)
        titles = extract_titles(cls_info, words)

        # y-axis scale
        y_min, y_max = extract_y_axis_range(im, cls_info)

        # bars
        bars = None
        if chart_type == "Bar":
            bar_db, bar_net, bar_test = methods["Bar"]
            bar_results = bar_test(im, bar_db, bar_net, debug=False)  # [tls, brs]
            tls_b, brs_b = bar_results[0], bar_results[1]
            bars = GroupBarRaw(im, tls_b, brs_b)

    return {
        "chart_type": chart_type,
        "chart_title_candidates": titles,
        "y_axis_min_est": y_min,
        "y_axis_max_est": y_max,
        "bars_raw": _to_py(bars),
    }
