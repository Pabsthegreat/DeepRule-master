import os
import base64

from django.conf import settings
from django.shortcuts import render
import torch.cuda

from test_pipe_type_cloud import Pre_load_nets, run_on_image

Lock = False
_CLOUD_METHODS = None
RESULT_SAVE_DIR = "output_fixed"


def _ensure_methods():
    global _CLOUD_METHODS
    if _CLOUD_METHODS is None:
        data_root = settings.BASE_DIR
        cache_dir = os.path.join(settings.BASE_DIR, "cache")
        _CLOUD_METHODS = Pre_load_nets("Bar", 0, data_root, cache_dir)
    return _CLOUD_METHODS


def _file_to_data_uri(path):
    with open(path, "rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _rescale_value(value, src_min, src_max, dst_min, dst_max):
    if value is None:
        return None
    if src_min is None or src_max is None:
        return value
    if abs(src_max - src_min) < 1e-6:
        return value
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def get_group(request):
    global Lock
    if Lock:
        return render(request, "onuse.html")

    if request.method != "POST":
        return render(request, "upload.html")

    print(f"The method is: {request.method}")
    if torch.cuda.is_available():
        print("Clean CUDA cache")
        torch.cuda.empty_cache()

    static_dir = os.path.join(settings.BASE_DIR, "static")
    os.makedirs(static_dir, exist_ok=True)
    target_path = os.path.join(static_dir, "target.png")
    save_dir = os.path.join(settings.BASE_DIR, RESULT_SAVE_DIR)

    Lock = True
    try:
        upload_file = request.FILES.get("file")
        if upload_file is not None:
            with open(target_path, "wb") as fout:
                for chunk in upload_file.chunks():
                    fout.write(chunk)
        if not os.path.exists(target_path):
            raise RuntimeError("No chart image uploaded.")

        min_field = request.POST.get("min", "").strip()
        max_field = request.POST.get("max", "").strip()
        override_min = override_max = None
        if min_field:
            try:
                override_min = float(min_field)
                override_max = float(max_field)
            except (TypeError, ValueError):
                override_min = override_max = None

        methods = _ensure_methods()
        result = run_on_image(
            target_path,
            "Bar",
            save_path=save_dir,
            methods_override=methods,
            return_images=True,
        )

        bars_summary = sorted(result.get("bars_summary") or [], key=lambda r: r.get("x1", 0))
        y_min_est = result.get("y_axis_min_est")
        y_max_est = result.get("y_axis_max_est")

        processed_rows = []
        summary_lines = []
        values = []

        for idx, row in enumerate(bars_summary, 1):
            value = row.get("value")
            if (
                value is not None
                and override_min is not None
                and override_max is not None
                and y_min_est is not None
                and y_max_est is not None
            ):
                value = _rescale_value(value, y_min_est, y_max_est, override_min, override_max)
            if value is not None:
                value = round(value, 2)
                values.append(value)

            category = row.get("category") or ""
            label = row.get("label") or ""
            color = row.get("color")

            processed_rows.append({
                "index": idx,
                "category": category,
                "label": label,
                "value": value,
                "color": color,
            })

            val_str = "--" if value is None else f"{value:,.2f}"
            if category and label:
                summary_lines.append(f"Bar {idx:02d}: {category} | {label} = {val_str}")
            elif category:
                summary_lines.append(f"Bar {idx:02d}: {category} = {val_str}")
            elif label:
                summary_lines.append(f"Bar {idx:02d}: {label} = {val_str}")
            else:
                summary_lines.append(f"Bar {idx:02d}: {val_str}")

        if override_min is not None and override_max is not None:
            y_axis_summary = f"{override_min:.2f} to {override_max:.2f}"
        elif y_min_est is not None and y_max_est is not None:
            y_axis_summary = f"{y_min_est:.2f} to {y_max_est:.2f}"
        else:
            y_axis_summary = "Not detected"

        titles = result.get("chart_title_candidates") or {}

        original_b64 = result.get("original_image_b64") or _file_to_data_uri(target_path)
        overlay_b64 = result.get("overlay_image_b64") or original_b64

        context = {
            "Type": result.get("chart_type", "Unknown"),
            "image": original_b64,
            "image_painted": overlay_b64,
            "min2max": y_axis_summary,
            "bar_rows": processed_rows,
            "bar_summary_lines": summary_lines,
            "ChartTitle": titles.get("2") or "None",
            "ValueAxisTitle": titles.get("1") or "None",
            "CategoryAxisTitle": titles.get("3") or "None",
        }

        if values:
            context["bar_stats"] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
            }

        if summary_lines:
            context["data"] = "\n".join(summary_lines)

        csv_path = result.get("csv_path")
        if csv_path:
            context["csv_path"] = os.path.relpath(csv_path, settings.BASE_DIR)

        return render(request, "results.html", context)
    finally:
        Lock = False
