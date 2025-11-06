#!/bin/bash
# Run Streamlit Demo Interface

echo "🚀 Starting DeepRule Streamlit Demo..."
echo ""
echo "📍 Access the app at: http://localhost:8501"
echo "📍 Django API at: http://localhost:8000/api/docs"
echo ""

source .venv/bin/activate
streamlit run streamlit_app.py
