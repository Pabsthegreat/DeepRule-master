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

        # Get chart type from form (default to Auto)
        chart_type = request.POST.get("chart_type", "Auto").strip()
        
        # Auto-detect chart type if needed
        data_root = settings.BASE_DIR
        cache_dir = os.path.join(settings.BASE_DIR, "cache")
        
        if chart_type == "Auto":
            from test_pipe_type_cloud import auto_detect_chart_type
            chart_type = auto_detect_chart_type(target_path, data_root, cache_dir)
            print(f"Auto-detected chart type: {chart_type}")
        
        if chart_type not in ["Bar", "Line", "Pie"]:
            chart_type = "Bar"
        
        min_field = request.POST.get("min", "").strip()
        max_field = request.POST.get("max", "").strip()
        override_min = override_max = None
        if min_field:
            try:
                override_min = float(min_field)
                override_max = float(max_field)
            except (TypeError, ValueError):
                override_min = override_max = None

        # Load appropriate model for chart type
        methods = Pre_load_nets(chart_type, 0, data_root, cache_dir)
        
        result = run_on_image(
            target_path,
            chart_type,
            save_path=save_dir,
            methods_override=methods,
            return_images=True,
        )

        processed_rows = []
        summary_lines = []
        values = []
        
        if chart_type == "Pie":
            # Handle pie chart results
            pie_summary = result.get("pie_summary") or []
            
            for idx, row in enumerate(pie_summary, 1):
                angle = row.get("angle_degrees")
                percentage = row.get("value")
                category = row.get("category") or ""
                
                processed_rows.append({
                    "index": idx,
                    "category": category,
                    "label": "",
                    "value": percentage,
                    "color": None,
                })
                
                if category:
                    summary_lines.append(f"Segment {idx:02d}: {category} = {percentage:.2f}%")
                else:
                    summary_lines.append(f"Segment {idx:02d}: {percentage:.2f}%")
                
                if percentage is not None:
                    values.append(percentage)
        
        elif chart_type == "Line":
            # Handle line chart results
            line_summary = result.get("lines_summary") or []
            
            for idx, row in enumerate(line_summary, 1):
                value = row.get("value")
                label = row.get("label") or ""
                x_pixel = row.get("x_pixel", 0)
                category = row.get("category") or ""
                color = row.get("color")
                
                processed_rows.append({
                    "index": idx,
                    "category": category,
                    "label": label,
                    "value": value,
                    "color": color,
                })
                
                # Format with category (X-axis) if available
                if category and label:
                    summary_lines.append(f"Point {idx:03d}: {category} | {label} → Y={value:.2f}")
                elif label:
                    summary_lines.append(f"Point {idx:03d}: {label} at X={x_pixel:.1f} → Y={value:.2f}")
                elif category:
                    summary_lines.append(f"Point {idx:03d}: {category} → Y={value:.2f}")
                else:
                    summary_lines.append(f"Point {idx:03d}: X={x_pixel:.1f} → Y={value:.2f}")
                
                if value is not None:
                    values.append(value)
        
        else:
            # Handle bar/line chart results
            bars_summary = sorted(result.get("bars_summary") or [], key=lambda r: r.get("x1", 0))
            y_min_est = result.get("y_axis_min_est")
            y_max_est = result.get("y_axis_max_est")

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

        if chart_type == "Pie":
            y_axis_summary = "N/A (Pie Chart)"
        elif chart_type == "Line" and override_min is not None and override_max is not None:
            y_axis_summary = f"{override_min:.2f} to {override_max:.2f}"
        elif override_min is not None and override_max is not None:
            y_axis_summary = f"{override_min:.2f} to {override_max:.2f}"
        elif result.get("y_axis_min_est") is not None and result.get("y_axis_max_est") is not None:
            y_axis_summary = f"{result.get('y_axis_min_est'):.2f} to {result.get('y_axis_max_est'):.2f}"
        else:
            y_axis_summary = "Not detected"

        titles = result.get("chart_title_candidates") or {}

        original_b64 = result.get("original_image_b64") or _file_to_data_uri(target_path)
        overlay_b64 = result.get("overlay_image_b64") or original_b64

        # Check if auto-detection was used
        was_auto_detected = request.POST.get("chart_type", "Auto").strip() == "Auto"
        
        context = {
            "Type": result.get("chart_type", "Unknown"),
            "auto_detected": was_auto_detected,
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
