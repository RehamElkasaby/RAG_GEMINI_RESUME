import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Optional
import json
from sentence_transformers import SentenceTransformer

from cv_schema import CVData

class VectorStore:
    """ChromaDB-based vector store for CV embeddings"""
    
    def __init__(self, collection_name: str = "cv_collection"):
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB client
        self.client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "CV embeddings for RAG system"}
            )
    
    def add_cv(self, cv_data: CVData) -> str:
        """Add a CV to the vector store"""
        try:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Convert CV to searchable text
            searchable_text = cv_data.to_searchable_text()
            
            # Generate embedding
            embedding = self.embedding_model.encode(searchable_text).tolist()
            
            # Prepare metadata
            metadata = {
                "filename": cv_data.filename,
                "candidate_name": cv_data.personal_info.name,
                "email": cv_data.personal_info.email,
                "total_experience_years": cv_data.get_total_experience_years(),
                "skills_count": len(cv_data.skills),
                "parsed_date": cv_data.parsed_date
            }
            
            # Add to collection
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[searchable_text],
                metadatas=[metadata]
            )
            
            # Store full CV data separately (as we can't store large objects in ChromaDB)
            self._store_cv_data(doc_id, cv_data.dict())
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Failed to add CV to vector store: {str(e)}")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant CVs based on query"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['metadatas', 'documents', 'distances']
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'id': results['ids'][0][i],
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                
                # Load full CV data
                cv_data = self._load_cv_data(result['id'])
                if cv_data:
                    result['cv_data'] = cv_data
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Failed to search vector store: {str(e)}")
    
    def get_all_cvs(self) -> List[Dict[str, Any]]:
        """Get all CVs in the vector store"""
        try:
            # Get all documents
            results = self.collection.get(
                include=['metadatas', 'documents']
            )
            
            formatted_results = []
            for i in range(len(results['ids'])):
                result = {
                    'id': results['ids'][i],
                    'document': results['documents'][i],
                    'metadata': results['metadatas'][i]
                }
                
                # Load full CV data
                cv_data = self._load_cv_data(result['id'])
                if cv_data:
                    result['cv_data'] = cv_data
                
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            raise Exception(f"Failed to get all CVs: {str(e)}")
    
    def delete_cv(self, doc_id: str) -> bool:
        """Delete a CV from the vector store"""
        try:
            self.collection.delete(ids=[doc_id])
            self._delete_cv_data(doc_id)
            return True
        except Exception as e:
            print(f"Failed to delete CV {doc_id}: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all CVs from the vector store"""
        try:
            # Delete collection
            self.client.delete_collection(name=self.collection_name)
            
            # Recreate collection
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "CV embeddings for RAG system"}
            )
            
            # Clear CV data storage
            import shutil
            import os
            if os.path.exists("./cv_data"):
                shutil.rmtree("./cv_data")
            
            return True
        except Exception as e:
            print(f"Failed to clear vector store: {str(e)}")
            return False
    
    def _store_cv_data(self, doc_id: str, cv_data: dict):
        """Store full CV data separately"""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs("./cv_data", exist_ok=True)
        
        # Store CV data as JSON file
        with open(f"./cv_data/{doc_id}.json", 'w') as f:
            json.dump(cv_data, f, indent=2)
    
    def _load_cv_data(self, doc_id: str) -> Optional[dict]:
        """Load full CV data"""
        try:
            with open(f"./cv_data/{doc_id}.json", 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _delete_cv_data(self, doc_id: str):
        """Delete CV data file"""
        import os
        try:
            os.remove(f"./cv_data/{doc_id}.json")
        except:
            pass
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            count = self.collection.count()
            return {
                "total_cvs": count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            return {"error": str(e)}
