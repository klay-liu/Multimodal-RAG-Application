# app/rag_engine.py
from typing import Optional
import asyncio


import os
import pymupdf
from tqdm import tqdm
from pathlib import Path
import base64
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from typing import List, Dict, Any, Optional
from openai import OpenAI

def pdf2imgs(pdf_path, pdf_pages_dir="pdf_pages"):
    """
    Convert a PDF file to individual PNG images for each page.

    Args:
        pdf_path (str): The path to the PDF file.
        pdf_pages_dir (str, optional): The directory to save the PNG images. Defaults to "pdf_pages".

    Returns:
        str: The path to the directory containing the PNG images.
    """
    import pypdfium2 as pdfium
    # Open the PDF document
    pdf = pdfium.PdfDocument(pdf_path)

    # Create the directory to save the PNG images if it doesn't exist
    os.makedirs(pdf_pages_dir, exist_ok=True)

    # Get the resolution of the first page to determine the scale factor
    resolution = pdf.get_page(0).render().to_numpy().shape
    scale = 1 if max(resolution) >= 1620 else 300 / 72  # Scale factor based on resolution

    # Get the number of pages in the PDF
    n_pages = len(pdf)

    # Loop through each page and save as a PNG image
    for page_number in range(n_pages):
        page = pdf.get_page(page_number)
        pil_image = page.render(
            scale=scale,
            rotation=0,
            crop=(0, 0, 0, 0),
            may_draw_forms=False,
            fill_color=(255, 255, 255, 255),
            draw_annots=False,
            grayscale=False,
        ).to_pil()
        image_path = os.path.join(pdf_pages_dir, f"{str(pdf_path).split('/')[-1]}_page_{page_number:03d}.png")
        pil_image.save(image_path)

    return pdf_pages_dir
                   

async def process_pdf(file_path: str) -> bool:
    """
    Process uploaded PDF file and initialize RAG system
    """
    try:
        # uploads/transformers_paper.pdf
        filename = Path(file_path.split("/")[-1]) 
        doc = pymupdf.open(file_path)
        num_pages = len(doc)

        # Define the directories to store the extracted text, images and page images from each page
        data_dir = Path(file_path.split("/")[0])
        image_save_dir = data_dir / "images"
        text_save_dir =data_dir / "text"
        page_images_save_dir = data_dir / "page_images"

        # Chunk the text for effective retrieval
        chunk_size = 1000
        overlap=100
        
        items = []
        # Process all pages of the PDF
        for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
            page = page = doc[page_num]
            text = page.get_text()
            
            # # Process chunks with overlap
            # chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]
            # !pip install -qU langchain-text-splitters 

            # Process chunks with RecursiveCharaterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                separators=[
                    "\n\n",
                    "\n",
                    " ",
                    ".",
                    ",",
                    "。",
                    "．",
                    "、",
                    "，",
                    "\u200b",  # 零宽度空格
                ],
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
            chunks = text_splitter.split_text(text)

            # Generate an item to add to items
            for i,chunk in enumerate(chunks):
                text_file_name = f"{text_save_dir}/{filename}_text_{page_num}_{i}.txt"
                # If the text folder doesn't exist, create one
                os.makedirs(text_save_dir, exist_ok=True)
                with open(text_file_name, 'w') as f:
                    f.write(chunk)
                
                item={}
                item["page"] = page_num
                item["type"] = "text"
                item["text"] = chunk
                item["path"] = text_file_name
                items.append(item)
            
            
            # Get all the images in the current page
            images = page.get_images()
            for idx, image in enumerate(images):        
                # Extract the image data
                xref = image[0]
                pix = pymupdf.Pixmap(doc, xref)
                pix.tobytes("png")
                # Create the image_name that includes the image path
                image_name = f"{image_save_dir}/{filename}_image_{page_num}_{idx}_{xref}.png"
                # If the image folder doesn't exist, create one
                os.makedirs(image_save_dir, exist_ok=True)
                # Save the image
                pix.save(image_name)
                
                # Produce base64 string
                with open(image_name, 'rb') as f:
                    image = base64.b64encode(f.read()).decode('utf8')
                
                item={}
                item["page"] = page_num
                item["type"] = "image"
                item["path"] = image_name
                item["image"] = image
                items.append(item)

        # Save pdf pages as images
        try:
            page_images_save_dir = pdf2imgs(file_path, page_images_save_dir)
        except Exception as e:
            print(f"Error in processing page image saving.")

        for page_num in range(num_pages):
            page_path = os.path.join(page_images_save_dir,  f"{str(file_path).split('/')[-1]}_page_{page_num:03d}.png")
            # Produce base64 string
            with open(page_path, 'rb') as f:
                page_image = base64.b64encode(f.read()).decode('utf8')
            
            item = {}
            item["page"] = page_num
            item["type"] = "page"
            item["path"] = page_path
            item["image"] = page_image
            items.append(item)
            
        return items
    except Exception as e:
        print(f"Error processing PDF: {str(e)}")
        return None

# class BGEVisualizedEncoder:
#     """Manager for BGE Visualized encoders with language support.
    
#     Supported languages:
#         - 'en': English model (bge-base-en-v1.5)
#         - 'multilingual': Multilingual model (bge-m3)
#     """
    
#     def __init__(self):
#         """Initialize English and multilingual encoders."""
#         try:
#             import sys
#             sys.path.append('../FlagEmbedding/research/visual_bge')
#             from visual_bge.modeling import Visualized_BGE
            
#             self.en_encoder = Visualized_BGE(
#                 model_name_bge="BAAI/bge-base-en-v1.5",
#                 model_weight="../models/Visualized_base_en_v1.5.pth"
#             ).eval()
            
#             self.m3_encoder = Visualized_BGE(
#                 model_name_bge="BAAI/bge-m3",
#                 model_weight="../models/Visualized_m3.pth"
#             ).eval()
#         except Exception as e:
#             raise RuntimeError(f"Failed to initialize encoders: {e}")

#     def get_encoder(self, language="en"):
#         """Get appropriate encoder based on language.
        
#         Args:
#             language (str): Language option ('en' or 'multilingual')
            
#         Returns:
#             Visualized_BGE: Initialized encoder model
#         """
#         return self.en_encoder if language == "en" else self.m3_encoder

class BGEVisualizedEncoder:
    """Manager for BGE Visualized encoders with language support."""
    
    def __init__(self, models_dir="models"):
        """Initialize encoders using local model files.
        
        Args:
            models_dir (str): Path to directory containing model files
        """
        try:
            # Convert to absolute path and verify models directory exists
            self.models_dir = Path(models_dir).resolve()
            if not self.models_dir.exists():
                raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

            # Check for required model files
            self.en_model_path = self.models_dir / "Visualized_base_en_v1.5.pth"
            self.m3_model_path = self.models_dir / "Visualized_m3.pth"

            if not self.en_model_path.exists():
                raise FileNotFoundError(f"English model not found at: {self.en_model_path}")
            if not self.m3_model_path.exists():
                raise FileNotFoundError(f"Multilingual model not found at: {self.m3_model_path}")

            # Import and initialize models
            import sys
            sys.path.append(Path('../FlagEmbedding/research/visual_bge').resolve())
            from visual_bge.modeling import Visualized_BGE
            
            self.en_encoder = Visualized_BGE(
                model_name_bge="BAAI/bge-base-en-v1.5",
                model_weight=str(self.en_model_path)
            ).eval()
            
            self.m3_encoder = Visualized_BGE(
                model_name_bge="BAAI/bge-m3",
                model_weight=str(self.m3_model_path)
            ).eval()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize encoders: {e}")

    def get_encoder(self, language="en"):
        """Get appropriate encoder based on language."""
        return self.en_encoder if language == "en" else self.m3_encoder


def generate_bge_visualized_embeddings(encoder, image_path=None, text=None):
    """Generate embeddings using provided encoder.
    
    Args:
        encoder: Initialized Visualized_BGE model
        image_path (str, optional): Path to input image
        text (str, optional): Text to encode with image
        
    Returns:
        list: Generated embeddings
        
    Raises:
        ValueError: If image_path is not provided
    """
    
    if not image_path and not text:
        raise ValueError("Image path or text must be provided")
        
    if image_path:
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
    try:
        if image_path and text:                
            return encoder.encode(image=image_path, text=text).tolist()[0]
        if text:
            return encoder.encode(text=text).tolist()[0]

        return encoder.encode(image=image_path).tolist()[0]
    
    except Exception as e:
        raise RuntimeError(f"Encoding failed: {e}")


class MilvusManager:
    """Manager class for Milvus vector database operations"""
    
    def __init__(self, uri: str = 'db/multimodal_rag_milvus_project.db'):
        """
        Initialize Milvus client
        
        Args:
            uri (str): URI for Milvus connection
        """
        self.client = MilvusClient(uri=uri)
        
    def create_collection(self, 
                         collection_name: str,
                         dimension: int,
                         recreate: bool = False) -> None:
        """
        Create a new collection in Milvus
        
        Args:
            collection_name (str): Name of the collection
            dimension (int): Dimension of vectors
            recreate (bool): Whether to drop existing collection and recreate
        """
        # Original code snippet
        if recreate and collection_name in self.client.list_collections():
            self.client.drop_collection(collection_name)
            
        if collection_name not in self.client.list_collections():
            self.client.create_collection(
                collection_name=collection_name,
                auto_id=True,
                dimension=dimension,
                enable_dynamic_field=True,
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")

    def insert_data(self, 
                    collection_name: str, 
                    items: List[Dict[str, Any]]) -> Dict:
        """
        Insert data into collection
        
        Args:
            collection_name (str): Name of the collection
            items (List[Dict]): List of items to insert
            
        Returns:
            Dict: Insertion result
        """
        try:
            result = self.client.insert(
                collection_name=collection_name,
                data=items
            )
            print(f"Successfully inserted {len(items)} items")
            return result
        except Exception as e:
            print(f"Error inserting data: {e}")
            return None

    def search(self,
               collection_name: str,
               query_embedding: List[float],
               output_fields: List[str],
               limit: int = 3) -> List[Dict]:
        """
        Search for similar vectors in collection
        
        Args:
            collection_name (str): Name of the collection
            query_embedding (List[float]): Query vector
            output_fields (List[str]): Fields to return in results
            limit (int): Number of results to return
            
        Returns:
            List[Dict]: Search results
        """
        # Original code snippet
        try:
            search_results = self.client.search(
                collection_name=collection_name,
                data=[query_embedding],
                output_fields=output_fields,
                search_params={
                    "metric_type": "COSINE",
                    "params": {}
                },
                limit=limit
            )[0]
            
            return [hit["entity"] for hit in search_results]
        except Exception as e:
            print(f"Error during search: {e}")
            return []

def generate_rag_response(prompt, matched_items):
    
    # Create context
    text_context = ""
    image_context = []
    
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    
    
    for item in matched_items:
        if 'text' in item.keys(): 
            text_context += str(item["page"]) + ". " + item['text'] + "\n"
        else:
            image_context.append(item['image'])
    
    final_prompt = f"""You are a helpful assistant for question answering.
    The text context is relevant information retrieved.
    The provided image(s) are relevant information retrieved.
    
    <context>
    {text_context}
    </context>
    
    Answer the following question using the relevant context and images.
    
    <question>
    {prompt}
    </question>
    
    Answer:"""
    if image_context:
        response = client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_context}"},
                        },
                    ],
                }
            ],
            max_tokens=500 # 根据需要调整
        )
    else:
        response = client.chat.completions.create(
            model="gemma-3-27b-it",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": final_prompt},
                    ],
                }
            ],
            max_tokens=500 # 根据需要调整
        )
    result = response.choices[0].message.content

    return result
    

# async def query_rag_system(message: str, image_path: Optional[str] = None, file_name: Optional[str] = None) -> str:
#     """
#     Query the RAG system with user message and optional image
#     """
#     try:
#         # 1. process pdf
#         items = await process_pdf(file_name)
        
#         # 2. generate embedding for pdf file
#         bge_encoder = BGEVisualizedEncoder()
#         encoder = bge_encoder.get_encoder(language="multilingual")
        
#         for item in tqdm(items, "Generating embeddings"):
#             if item['type'] == 'text':
#                 item['vector'] = generate_bge_visualized_embeddings(encoder=encoder, text=item['text'])
#             else:
#                 item['vector'] = generate_bge_visualized_embeddings(encoder=encoder, image_path=item['path'])
        
#         # 3. set up Milvus database
#         milvus_mgr = MilvusManager()
#         dim = len(items[0]['vector'])
#         collection_name = "multimodal_rag_on_pdf"

#         milvus_mgr.create_collection(
#             collection_name=collection_name,
#             dimension=dim,
#             recreate=False
#         )

#         insert_result = milvus_mgr.insert_data(
#             collection_name=collection_name,
#             items=items
#         )

#         # Generate query embedding using both text and image if available
#         query_embedding = generate_bge_visualized_embeddings(
#             encoder=encoder,
#             text=message,
#             image_path=image_path if image_path else None
#         )

#         # Search in Milvus
#         matched_items = milvus_mgr.search(
#             collection_name=collection_name,
#             query_embedding=query_embedding,
#             output_fields=['text', 'page', 'image'],
#             limit=3
#         )

#         # Generate response
#         results = generate_rag_response(message, matched_items)

#         return results
#     except Exception as e:
#         return f"Error generating response: {str(e)}"

# app/rag_engine.py
async def query_rag_system(message: str, image_path: Optional[str] = None, file_name: Optional[str] = None) -> str:
    """
    Query the RAG system with user message and optional image
    """
    try:
        # 1. Initialize encoder
        bge_encoder = BGEVisualizedEncoder("models")
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