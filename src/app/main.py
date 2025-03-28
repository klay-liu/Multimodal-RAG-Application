# src/app/main.py
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pathlib import Path

from app.routers import upload, chat

app = FastAPI()

# src/app/main.py
from pathlib import Path


# --- Define the Project Root Explicitly ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "db" / "multimodal_rag_milvus_project.db"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMP_DIR = Path(__file__).resolve().parent / "temp"  # Keep temp relative to app
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

app.include_router(upload.router)
app.include_router(chat.router)

@app.on_event("startup")
async def startup_event():
    """Verify required directories and models exist on startup."""
    try:
        # Create required directories
        UPLOAD_DIR.mkdir(exist_ok=True)
        TEMP_DIR.mkdir(exist_ok=True)
        
        # Verify models exist
        required_models = [
            "Visualized_base_en_v1.5.pth",
            "Visualized_m3.pth"
        ]
        
        missing_models = [
            model for model in required_models 
            if not (MODELS_DIR / model).exists()
        ]

        if missing_models:
            raise FileNotFoundError(
                f"Missing required models in {MODELS_DIR}: {', '.join(missing_models)}"
            )
            
    except Exception as e:
        raise RuntimeError(f"Failed to initialize application: {e}")

@app.get("/check-setup")
async def check_setup():
    """Check if all required files and models are present"""
    try:
        required_models = [
            "Visualized_base_en_v1.5.pth",
            "Visualized_m3.pth"
        ]
        
        status = {
            "models_dir": str(MODELS_DIR),
            "models_exist": all((MODELS_DIR / model).exists() for model in required_models),
            "upload_dir": str(UPLOAD_DIR),
            "upload_dir_exists": UPLOAD_DIR.exists(),
            "temp_dir": str(TEMP_DIR),
            "temp_dir_exists": TEMP_DIR.exists()
        }
        
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
