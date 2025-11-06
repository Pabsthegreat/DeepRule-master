# Interface Comparison Guide

## Overview

DeepRule now offers **three ways** to extract data from charts:

| Interface | Best For | Port | Access |
|-----------|----------|------|--------|
| **Django Web UI** | Production, sharing with others | 8000 | http://localhost:8000 |
| **Streamlit Demo** | Quick testing, internal demos | 8501 | http://localhost:8501 |
| **REST API** | Programmatic access, integrations | 8000 | http://localhost:8000/api/extract |

## 1. Django Web UI ⭐ (Recommended for Production)

### Launch
```bash
source .venv/bin/activate
python manage.py runserver 0.0.0.0:8000
```

### Features
✅ Professional, modern UI with gradient design
✅ RESTful API endpoint included
✅ CORS enabled for cross-origin requests
✅ Production-ready with gunicorn/uwsgi
✅ Session management and CSRF protection
✅ Concurrent user support
✅ Custom URL routing
✅ Static file serving

### Use Cases
- Sharing with external users
- Production deployment
- When you need API + UI together
- Corporate environments
- Public-facing service

### Screenshots
- Modern drag-and-drop interface
- Real-time upload status
- Professional results display
- Downloadable CSV export

---

## 2. Streamlit Demo UI 🎯 (Great for Quick Testing)

### Launch
```bash
./run_streamlit.sh
# or
streamlit run streamlit_app.py
```

### Features
✅ Ultra-simple Python-only code
✅ Interactive widgets and controls
✅ Real-time model caching status
✅ Side-by-side image comparison
✅ Inline data tables with pandas
✅ Instant CSV download
✅ Auto-refresh on code changes

### Use Cases
- Quick internal demos
- Rapid prototyping
- Data science presentations
- Testing new features
- Stakeholder demos

### Limitations
❌ No built-in REST API
❌ Single-threaded (slower with multiple users)
❌ Session state can be tricky
❌ Less suitable for production

---

## 3. REST API 🔌 (For Integrations)

### Endpoint
```
POST http://localhost:8000/api/extract
```

### Features
✅ Language-agnostic (curl, Python, JavaScript, etc.)
✅ JSON responses
✅ CORS enabled
✅ Comprehensive error handling
✅ Metadata included (titles, ranges, etc.)
✅ Debug mode support

### Use Cases
- Integration with other applications
- Batch processing scripts
- Mobile app backends
- Automated workflows
- Third-party integrations

### Documentation
http://localhost:8000/api/docs

---

## Feature Comparison Matrix

| Feature | Django | Streamlit | API |
|---------|--------|-----------|-----|
| Chart Type Auto-detect | ✅ | ✅ | ✅ |
| Bar Charts | ✅ | ✅ | ✅ |
| Line Charts | ✅ | ✅ | ✅ |
| Pie Charts | ✅ | ✅ | ✅ |
| Y-axis Rescaling | ✅ | ✅ | ✅ |
| Debug Mode | ✅ | ✅ | ✅ |
| CSV Export | ✅ | ✅ | ✅ |
| REST API | ✅ | ❌ | ✅ |
| Drag & Drop | ✅ | ✅ | N/A |
| Model Caching | ✅ | ✅ | ✅ |
| Concurrent Users | ✅✅✅ | ⚠️ | ✅✅✅ |
| Production Ready | ✅✅✅ | ⚠️ | ✅✅✅ |
| Setup Complexity | Medium | Easy | Easy |
| Code Maintenance | Medium | Easy | N/A |

Legend:
- ✅✅✅ = Excellent
- ✅ = Good
- ⚠️ = Limited
- ❌ = Not Available
- N/A = Not Applicable

---

## Recommendations by Scenario

### Scenario 1: "I want to share this with clients"
**Use Django** - Professional UI, production-ready, has API

### Scenario 2: "I need to demo this to my team quickly"
**Use Streamlit** - Fastest setup, interactive, looks great

### Scenario 3: "I want to integrate with my Python script"
**Use API** - Programmatic access, easy integration

### Scenario 4: "I need to process 1000 images"
**Use API** - Write a Python loop calling the API

### Scenario 5: "I'm still developing/testing"
**Use Streamlit** - Fast iteration, easy debugging

### Scenario 6: "We need this in production with 100+ users"
**Use Django** - Scalable, concurrent, production-tested

---

## Running Multiple Interfaces Simultaneously

You can run all three at the same time!

**Terminal 1 - Django:**
```bash
source .venv/bin/activate
python manage.py runserver 0.0.0.0:8000
```

**Terminal 2 - Streamlit:**
```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Now you have:
- Django UI: http://localhost:8000
- API Docs: http://localhost:8000/api/docs
- Streamlit: http://localhost:8501

---

## Performance Comparison

| Metric | Django | Streamlit |
|--------|--------|-----------|
| Cold Start | ~2-3s | ~3-5s |
| Warm Start | ~0.5s | ~0.5s |
| Multiple Users | Excellent | Poor |
| Memory Usage | Low | Medium |
| CPU Usage | Low | Medium |

---

## Code Maintenance

### Django
- **Files**: `server_match/view.py`, `templates/*.html`, `server_match/urls.py`
- **Complexity**: Medium (need to know Django patterns)
- **Lines of Code**: ~500

### Streamlit
- **Files**: `streamlit_app.py`
- **Complexity**: Low (pure Python)
- **Lines of Code**: ~300

### API
- **Files**: Same as Django (`server_match/view.py`)
- **Complexity**: Low (just endpoints)
- **Lines of Code**: ~150

---

## Debug Mode Comparison

### Django
```bash
# Web UI automatically uses debug=false
# API supports debug parameter
curl -X POST http://localhost:8000/api/extract \
  -F "file=@chart.png" \
  -F "debug=true"
```

### Streamlit
```python
# Checkbox in sidebar
debug_mode = st.checkbox("Debug Mode")
```

Both save debug images to `debug_output/` folder when enabled.

---

## Which Should I Use?

### Use Django if:
- ✅ You need a production-ready solution
- ✅ You want API + UI in one package
- ✅ You expect multiple concurrent users
- ✅ You need CSRF protection and security
- ✅ You're comfortable with web frameworks

### Use Streamlit if:
- ✅ You want the fastest setup
- ✅ Internal demos and testing only
- ✅ You prefer pure Python (no HTML/CSS)
- ✅ You want rapid iteration
- ✅ Single-user scenarios

### Use API if:
- ✅ You're integrating with other apps
- ✅ You need programmatic access
- ✅ You're building a pipeline
- ✅ You want to use any programming language

---

## Summary

🎯 **For most users**: Start with **Django** (production-ready, has API)

🚀 **For quick demos**: Use **Streamlit** (easiest, fastest)

🔌 **For developers**: Use the **API** (flexible, programmable)

**Best approach**: Use Django for production, Streamlit for internal testing!
