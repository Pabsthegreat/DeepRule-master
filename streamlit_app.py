"""
Streamlit UI for DeepRule Chart Data Extraction
Quick demo interface for internal testing
"""

import streamlit as st
import tempfile
import os
import pandas as pd
from PIL import Image
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_pipe_type_cloud import Pre_load_nets, run_on_image, auto_detect_chart_type

# Page config
st.set_page_config(
    page_title="DeepRule - Chart Extraction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 4px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
    st.session_state.cached_models = {}

# Header
st.title("🎯 DeepRule Chart Extraction")
st.markdown("AI-powered data extraction from chart images")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    chart_type = st.selectbox(
        "Chart Type",
        ["Auto-detect", "Bar", "Line", "Pie"],
        help="Auto-detect works for most charts"
    )
    
    st.markdown("---")
    
    st.subheader("Y-Axis Rescaling (Optional)")
    use_rescale = st.checkbox("Enable Y-axis rescaling")
    
    min_val = None
    max_val = None
    if use_rescale:
        col1, col2 = st.columns(2)
        with col1:
            min_val = st.number_input("Min", value=0.0)
        with col2:
            max_val = st.number_input("Max", value=100.0)
    
    st.markdown("---")
    
    debug_mode = st.checkbox(
        "Debug Mode",
        help="Save intermediate X-axis extraction images"
    )
    
    st.markdown("---")
    
    st.subheader("📚 Resources")
    st.markdown("[API Documentation](http://localhost:8000/api/docs)")
    st.markdown("[Django UI](http://localhost:8000)")

# Main content
uploaded_file = st.file_uploader(
    "Upload Chart Image",
    type=["png", "jpg", "jpeg"],
    help="Drag and drop or click to upload"
)

if uploaded_file is not None:
    # Display uploaded image
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Original Image")
        image = Image.open(uploaded_file)
        st.image(image, width='stretch')
    
    # Process button
    if st.button("🚀 Extract Data", type="primary"):
        with st.spinner("Processing chart..."):
            try:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Determine chart type
                actual_chart_type = chart_type
                if chart_type == "Auto-detect":
                    with st.spinner("Auto-detecting chart type..."):
                        actual_chart_type = auto_detect_chart_type(
                            tmp_path,
                            data_dir=".",
                            cache_dir="./cache"
                        )
                        st.info(f"🤖 Detected: **{actual_chart_type} Chart**")
                else:
                    actual_chart_type = chart_type
                
                # Load models if not cached
                if actual_chart_type not in st.session_state.cached_models:
                    with st.spinner(f"Loading {actual_chart_type} models..."):
                        methods = Pre_load_nets(
                            actual_chart_type,
                            id_cuda=0,
                            data_dir=".",
                            cache_dir="./cache"
                        )
                        st.session_state.cached_models[actual_chart_type] = methods
                        st.success(f"✅ {actual_chart_type} models loaded")
                
                # Run extraction
                methods = st.session_state.cached_models[actual_chart_type]
                
                result = run_on_image(
                    tmp_path,
                    actual_chart_type,
                    save_path="output_fixed",
                    methods_override=methods,
                    return_images=True,
                    debug=debug_mode
                )
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                # Display results
                with col2:
                    st.subheader("🎨 Detected Elements")
                    if result.get("overlay_image_b64"):
                        import base64
                        import io
                        img_data = base64.b64decode(result["overlay_image_b64"].split(",")[1])
                        overlay_img = Image.open(io.BytesIO(img_data))
                        st.image(overlay_img, width='stretch')
                
                st.markdown("---")
                
                # Summary
                st.subheader("📋 Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                
                with summary_col1:
                    st.metric("Chart Type", result.get("chart_type", "Unknown"))
                
                # Display data based on chart type
                if actual_chart_type == "Pie":
                    pie_data = result.get("pie_summary", [])
                    with summary_col2:
                        st.metric("Segments", len(pie_data))
                    with summary_col3:
                        if pie_data:
                            total = sum(d.get("value", 0) for d in pie_data)
                            st.metric("Total %", f"{total:.1f}%")
                    
                    # Data table
                    st.subheader("🥧 Pie Chart Data")
                    if pie_data:
                        df = pd.DataFrame([
                            {
                                "Category": d.get("category", ""),
                                "Percentage": f"{d.get('value', 0):.2f}%",
                                "Angle": f"{d.get('angle_degrees', 0):.1f}°"
                            }
                            for d in pie_data
                        ])
                        st.dataframe(df, use_container_width=True)
                
                elif actual_chart_type == "Line":
                    line_data = result.get("lines_summary", [])
                    with summary_col2:
                        st.metric("Data Points", len(line_data))
                    
                    y_min = result.get("y_axis_min_est")
                    y_max = result.get("y_axis_max_est")
                    with summary_col3:
                        if y_min is not None and y_max is not None:
                            st.metric("Y Range", f"{y_min:.1f} - {y_max:.1f}")
                    
                    # Data table
                    st.subheader("📈 Line Chart Data")
                    if line_data:
                        df = pd.DataFrame([
                            {
                                "Category": d.get("category", ""),
                                "Label": d.get("label", ""),
                                "Value": f"{d.get('value', 0):.2f}" if d.get('value') else "N/A",
                                "Color": d.get("color", "")
                            }
                            for d in line_data
                        ])
                        st.dataframe(df, use_container_width=True)
                
                else:  # Bar
                    bar_data = result.get("bars_summary", [])
                    with summary_col2:
                        st.metric("Bars Detected", len(bar_data))
                    
                    y_min = result.get("y_axis_min_est")
                    y_max = result.get("y_axis_max_est")
                    with summary_col3:
                        if y_min is not None and y_max is not None:
                            st.metric("Y Range", f"{y_min:.1f} - {y_max:.1f}")
                    
                    # Data table
                    st.subheader("📊 Bar Chart Data")
                    if bar_data:
                        # Apply rescaling if enabled
                        processed_data = []
                        for d in bar_data:
                            value = d.get("value")
                            if value is not None and use_rescale and min_val is not None and max_val is not None:
                                if y_min is not None and y_max is not None:
                                    value = min_val + (value - y_min) * (max_val - min_val) / (y_max - y_min)
                            
                            processed_data.append({
                                "Category": d.get("category", ""),
                                "Label": d.get("label", ""),
                                "Value": f"{value:.2f}" if value is not None else "N/A",
                                "Color": d.get("color", "")
                            })
                        
                        df = pd.DataFrame(processed_data)
                        st.dataframe(df, use_container_width=True)
                
                # Titles
                titles = result.get("chart_title_candidates", {})
                if titles:
                    st.subheader("📝 Detected Titles")
                    title_cols = st.columns(3)
                    
                    with title_cols[0]:
                        chart_title = titles.get("2")
                        if chart_title and chart_title != "None":
                            st.markdown(f"**Chart:** {chart_title}")
                    
                    with title_cols[1]:
                        value_title = titles.get("1")
                        if value_title and value_title != "None":
                            st.markdown(f"**Value Axis:** {value_title}")
                    
                    with title_cols[2]:
                        cat_title = titles.get("3")
                        if cat_title and cat_title != "None":
                            st.markdown(f"**Category Axis:** {cat_title}")
                
                # CSV download
                csv_path = result.get("csv_path")
                if csv_path and os.path.exists(csv_path):
                    st.markdown("---")
                    with open(csv_path, "r") as f:
                        csv_data = f.read()
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=os.path.basename(csv_path),
                        mime="text/csv"
                    )
                
                # Debug info
                if debug_mode:
                    st.markdown("---")
                    st.info("🐛 Debug mode enabled. Check `debug_output/` folder for intermediate images.")
                
                st.success("✅ Extraction completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("See error details"):
                    st.code(traceback.format_exc())

else:
    # Landing page
    st.info("👆 Upload a chart image to get started")
    
    st.markdown("---")
    
    # Features
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 📊 Multi-Chart Support
        - Bar Charts
        - Line Charts  
        - Pie Charts
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI-Powered
        - Auto chart detection
        - OCR for labels
        - Multi-orientation support
        """)
    
    with col3:
        st.markdown("""
        ### 🔧 Flexible
        - Debug mode
        - Y-axis rescaling
        - CSV export
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 14px;'>
    <p>DeepRule Chart Extraction • Streamlit Demo Interface</p>
    <p>For production use, see the <a href='http://localhost:8000/api/docs' target='_blank'>API documentation</a></p>
</div>
""", unsafe_allow_html=True)
