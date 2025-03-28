# src/app/routers/upload.py
from fastapi import APIRouter, File, UploadFile
import shutil
from pathlib import Path
from app.rag_engine import process_pdf, BGEVisualizedEncoder, generate_bge_visualized_embeddings, MilvusManager
from tqdm import tqdm

router = APIRouter(prefix="/upload", tags=["upload"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "db" / "multimodal_rag_milvus_project.db"
STATIC_DIR = PROJECT_ROOT / "static"
TEMPLATES_DIR = PROJECT_ROOT / "templates"
TEMP_DIR = Path(__file__).resolve().parent / "temp"  # Keep temp relative to app
UPLOAD_DIR = PROJECT_ROOT / "uploads"

@router.post("/")
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