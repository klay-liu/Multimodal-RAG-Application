# app/main.py
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import Optional
import shutil
import os
from pathlib import Path
from .rag_engine import process_pdf, query_rag_system, BGEVisualizedEncoder, generate_bge_visualized_embeddings, MilvusManager
from tqdm import tqdm


app = FastAPI()

# Configure paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR.parent/ "models"  # points to ../models/
UPLOAD_DIR = BASE_DIR / "uploads"
TEMP_DIR = BASE_DIR / "temp"
UPLOAD_DIR.mkdir(exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

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
        print(f"Startup Error: {e}")
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


# @app.post("/upload")
# async def upload_file(file: UploadFile = File(...)):
#     try:
#         file_path = UPLOAD_DIR / file.filename
#         with file_path.open("wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         # Process the PDF and initialize RAG system
#         success = await process_pdf(str(file_path))
        
#         if success:
#             return {"message": "File uploaded and processed successfully"}
#         return {"message": "Error processing file"}
#     except Exception as e:
#         return {"message": f"Error: {str(e)}"}

# app/main.py
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        # 1. Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Process PDF and store in Milvus
        items = await process_pdf(str(file_path))
        
        # 3. Generate embeddings
        bge_encoder = BGEVisualizedEncoder(models_dir=str(MODELS_DIR))
        encoder = bge_encoder.get_encoder(language="multilingual")
        
        for item in tqdm(items, "Generating embeddings"):
            if item['type'] == 'text':
                item['vector'] = generate_bge_visualized_embeddings(encoder=encoder, text=item['text'])
            else:
                item['vector'] = generate_bge_visualized_embeddings(encoder=encoder, image_path=item['path'])
        
        # 4. Store in Milvus
        milvus_mgr = MilvusManager()
        dim = len(items[0]['vector'])
        collection_name = "multimodal_rag_on_pdf"

        milvus_mgr.create_collection(
            collection_name=collection_name,
            dimension=dim,
            recreate=False
        )

        insert_result = milvus_mgr.insert_data(
            collection_name=collection_name,
            items=items
        )
        
        return {"message": "File uploaded and processed successfully", "filename": file.filename}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}

@app.post("/chat")
async def chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    file_name: Optional[str] = Form(None)
):
    try:
        # Handle image if provided
        image_path = None
        if image:
            # Save the uploaded image temporarily
            temp_image_path = f"temp/{image.filename}"
            with open(temp_image_path, "wb") as buffer:
                content = await image.read()
                buffer.write(content)
            image_path = temp_image_path

        response = await query_rag_system(message, image_path, file_name)
        return {"response": response}
    except Exception as e:
        return {"message": f"Error: {str(e)}"}
    
    