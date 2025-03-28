# Manages Milvus connection and operations
from pymilvus import MilvusClient
from typing import List, Dict, Any
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "db" / "multimodal_rag_milvus_project.db"



class MilvusManager:
    """Manager class for Milvus vector database operations"""
    
    def __init__(self, uri: str = str(DB_PATH)):
        """
        Initialize Milvus client
        
        Args:
            uri (str): URI for Milvus connection
        """
        # Create the database directory if it doesn't exist
        DB_DIR = DB_PATH.parent
        DB_DIR.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = MilvusClient(uri=uri)
            print(f"Successfully connected to Milvus at {uri}")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Milvus: {e}")
        
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
        try:
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
        except Exception as e:
            raise Exception(f"Failed to create collection: {e}")
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

    def __del__(self):
        """Cleanup when the instance is destroyed"""
        try:
            if hasattr(self, 'client'):
                self.client.close()
        except Exception as e:
            print(f"Error closing Milvus connection: {e}")
