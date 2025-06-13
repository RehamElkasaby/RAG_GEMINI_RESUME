import json
import os
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import math

from cv_schema_simple import CVData

class SimpleVectorStore:
    """Simple in-memory vector store for CV embeddings without external dependencies"""
    
    def __init__(self, storage_dir: str = "./cv_storage"):
        self.storage_dir = storage_dir
        self.collection_file = os.path.join(storage_dir, "collection.json")
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        
        # Initialize collection
        self.collection = self._load_collection()
    
    def _load_collection(self) -> Dict[str, Any]:
        """Load collection from file"""
        if os.path.exists(self.collection_file):
            try:
                with open(self.collection_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        # Return empty collection
        return {
            "documents": {},
            "metadata": {},
            "created_at": datetime.now().isoformat()
        }
    
    def _save_collection(self):
        """Save collection to file"""
        try:
            with open(self.collection_file, 'w') as f:
                json.dump(self.collection, f, indent=2)
        except Exception as e:
            print(f"Error saving collection: {str(e)}")
    
    def add_cv(self, cv_data: CVData) -> str:
        """Add a CV to the vector store"""
        try:
            # Generate unique ID
            doc_id = str(uuid.uuid4())
            
            # Convert CV to searchable text
            searchable_text = cv_data.to_searchable_text()
            
            # Prepare metadata
            metadata = {
                "filename": cv_data.filename,
                "candidate_name": cv_data.personal_info.name,
                "email": cv_data.personal_info.email,
                "total_experience_years": cv_data.get_total_experience_years(),
                "skills_count": len(cv_data.skills),
                "parsed_date": cv_data.parsed_date,
                "added_at": datetime.now().isoformat()
            }
            
            # Store document and metadata
            self.collection["documents"][doc_id] = searchable_text
            self.collection["metadata"][doc_id] = metadata
            
            # Store full CV data separately
            self._store_cv_data(doc_id, cv_data.dict())
            
            # Save collection
            self._save_collection()
            
            return doc_id
            
        except Exception as e:
            raise Exception(f"Failed to add CV to vector store: {str(e)}")
    
    def search(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant CVs based on text similarity"""
        try:
            query_lower = query.lower()
            results = []
            
            # Simple text-based search
            for doc_id, document in self.collection["documents"].items():
                # Calculate similarity score based on keyword matching
                similarity = self._calculate_text_similarity(query_lower, document.lower())
                
                if similarity > 0:  # Only include results with some similarity
                    result = {
                        'id': doc_id,
                        'document': document,
                        'metadata': self.collection["metadata"].get(doc_id, {}),
                        'similarity': similarity
                    }
                    
                    # Load full CV data
                    cv_data = self._load_cv_data(doc_id)
                    if cv_data:
                        result['cv_data'] = cv_data
                    
                    results.append(result)
            
            # Sort by similarity and return top N
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:n_results]
            
        except Exception as e:
            raise Exception(f"Failed to search vector store: {str(e)}")
    
    def _calculate_text_similarity(self, query: str, document: str) -> float:
        """Calculate simple text similarity based on keyword matching"""
        query_words = set(query.split())
        doc_words = set(document.split())
        
        if not query_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = query_words.intersection(doc_words)
        union = query_words.union(doc_words)
        
        if not union:
            return 0.0
        
        jaccard_sim = len(intersection) / len(union)
        
        # Boost score for exact phrase matches
        phrase_bonus = 0.0
        if query in document:
            phrase_bonus = 0.3
        
        # Boost score for multiple word matches
        word_match_ratio = len(intersection) / len(query_words)
        
        return min(1.0, jaccard_sim + phrase_bonus + (word_match_ratio * 0.2))
    
    def get_all_cvs(self) -> List[Dict[str, Any]]:
        """Get all CVs in the vector store"""
        try:
            results = []
            
            for doc_id, document in self.collection["documents"].items():
                result = {
                    'id': doc_id,
                    'document': document,
                    'metadata': self.collection["metadata"].get(doc_id, {})
                }
                
                # Load full CV data
                cv_data = self._load_cv_data(doc_id)
                if cv_data:
                    result['cv_data'] = cv_data
                
                results.append(result)
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to get all CVs: {str(e)}")
    
    def delete_cv(self, doc_id: str) -> bool:
        """Delete a CV from the vector store"""
        try:
            # Remove from collection
            if doc_id in self.collection["documents"]:
                del self.collection["documents"][doc_id]
            if doc_id in self.collection["metadata"]:
                del self.collection["metadata"][doc_id]
            
            # Delete CV data file
            self._delete_cv_data(doc_id)
            
            # Save collection
            self._save_collection()
            
            return True
        except Exception as e:
            print(f"Failed to delete CV {doc_id}: {str(e)}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all CVs from the vector store"""
        try:
            # Clear collection
            self.collection = {
                "documents": {},
                "metadata": {},
                "created_at": datetime.now().isoformat()
            }
            
            # Save empty collection
            self._save_collection()
            
            # Clear CV data storage
            import shutil
            cv_data_dir = os.path.join(self.storage_dir, "cv_data")
            if os.path.exists(cv_data_dir):
                shutil.rmtree(cv_data_dir)
            
            return True
        except Exception as e:
            print(f"Failed to clear vector store: {str(e)}")
            return False
    
    def _store_cv_data(self, doc_id: str, cv_data: dict):
        """Store full CV data separately"""
        cv_data_dir = os.path.join(self.storage_dir, "cv_data")
        os.makedirs(cv_data_dir, exist_ok=True)
        
        # Store CV data as JSON file
        with open(os.path.join(cv_data_dir, f"{doc_id}.json"), 'w') as f:
            json.dump(cv_data, f, indent=2)
    
    def _load_cv_data(self, doc_id: str) -> Optional[dict]:
        """Load full CV data"""
        try:
            cv_data_path = os.path.join(self.storage_dir, "cv_data", f"{doc_id}.json")
            with open(cv_data_path, 'r') as f:
                return json.load(f)
        except:
            return None
    
    def _delete_cv_data(self, doc_id: str):
        """Delete CV data file"""
        try:
            cv_data_path = os.path.join(self.storage_dir, "cv_data", f"{doc_id}.json")
            if os.path.exists(cv_data_path):
                os.remove(cv_data_path)
        except:
            pass
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            total_cvs = len(self.collection["documents"])
            
            # Calculate some basic stats
            total_skills = 0
            total_experience_years = 0
            
            for metadata in self.collection["metadata"].values():
                total_skills += metadata.get("skills_count", 0)
                total_experience_years += metadata.get("total_experience_years", 0)
            
            avg_skills = total_skills / total_cvs if total_cvs > 0 else 0
            avg_experience = total_experience_years / total_cvs if total_cvs > 0 else 0
            
            return {
                "total_cvs": total_cvs,
                "average_skills_per_cv": round(avg_skills, 1),
                "average_experience_years": round(avg_experience, 1),
                "storage_directory": self.storage_dir,
                "created_at": self.collection.get("created_at", "Unknown")
            }
        except Exception as e:
            return {"error": str(e)}