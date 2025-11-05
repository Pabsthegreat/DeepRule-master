# app.py
import os, tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# import directly from DeepRule script
from test_pipe_type_cloud import Pre_load_nets, run_on_image

app = FastAPI(title="DeepRule ChartOCR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

METHODS = None
CHART_TYPE = "Bar"

@app.on_event("startup")
def load_models_once():
    global METHODS
    METHODS = Pre_load_nets(CHART_TYPE, 0, ".", "./cache")

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    if METHODS is None:
        raise HTTPException(500, "Models not loaded")
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        result = run_on_image(tmp_path, CHART_TYPE)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
