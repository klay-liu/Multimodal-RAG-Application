# Contains the BGE encoder and embedding generation logic
from pathlib import Path

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

            # Import and initialize model
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
