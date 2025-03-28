# Handles PDF to images, text and image extraction, chunking
import base64
import os
import pymupdf
from tqdm import tqdm
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

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
        print(f"file_path: {file_path}")
        # Define the directories to store the extracted text, images and page images from each page
        data_dir = Path(file_path).parent.parent / "data"
        print(f"pdfdata_dir: {data_dir}")
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
