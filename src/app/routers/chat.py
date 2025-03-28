# src/app/routers/chat.py
from fastapi import APIRouter, Form, UploadFile, File
from typing import Optional

from app.rag_engine import query_rag_system

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/")
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
    