# Orchestrates the RAG flow
from typing import Optional, List, Dict, Any
from pathlib import Path

from app.document_processing import process_pdf as pdf_processor
from app.embedding import BGEVisualizedEncoder, generate_bge_visualized_embeddings
from app.vector_database import MilvusManager
from app.response_generation import generate_rag_response

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


async def process_pdf(file_path: str) -> List[Dict[str, Any]]:
    return await pdf_processor(file_path)

async def query_rag_system(message: str, image_path: Optional[str] = None, file_name: Optional[str] = None) -> str:
    """
    Query the RAG system with user message and optional image
    """
    try:
        models_dir = MODELS_DIR
        # 1. Initialize encoder
        bge_encoder = BGEVisualizedEncoder(str(models_dir))
        encoder = bge_encoder.get_encoder(language="multilingual")

        # 2. Generate query embedding
        query_embedding = generate_bge_visualized_embeddings(
            encoder=encoder,
            text=message,
            image_path=image_path if image_path else None
        )

        # 3. Search in Milvus
        milvus_mgr = MilvusManager()
        collection_name = "multimodal_rag_on_pdf"

        matched_items = milvus_mgr.search(
            collection_name=collection_name,
            query_embedding=query_embedding,
            output_fields=['text', 'page', 'image'],
            limit=3
        )

        # 4. Generate response
        results = generate_rag_response(message, matched_items)

        return results
    except Exception as e:
        return f"Error generating response: {str(e)}"
