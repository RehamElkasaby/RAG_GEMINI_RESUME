import os
from typing import List, Dict, Any, Optional
from vector_store_simple import SimpleVectorStore
from cv_schema_simple import CVData

# Optional import for Google Generative AI
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

class GeminiRAGPipeline:
    """RAG pipeline for CV querying using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        if not GEMINI_AVAILABLE:
            raise ValueError("Google Generative AI package not available. Please install google-generativeai package.")
        
        # Initialize Gemini API
        self.api_key = api_key or os.getenv('GOOGLE_API_KEY')
        if not self.api_key:
            raise ValueError("Google API key is required. Please provide it or set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
        # Initialize vector store
        self.vector_store = SimpleVectorStore()
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self) -> str:
        """Create the prompt template for the RAG system"""
        return """You are a helpful AI assistant specializing in recruitment and CV analysis. 
You have access to a database of candidate CVs and can answer questions about them.

Context from CV database:
{context}

Question: {question}

Instructions:
1. Analyze the provided CV context carefully
2. Answer the recruiter's question based on the available information
3. Be specific and mention candidate names when relevant
4. If the information is not available in the context, say so clearly
5. Provide actionable insights for recruitment decisions
6. Format your response in a clear, professional manner

Answer:"""
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question"""
        try:
            # Search for relevant CVs
            search_results = self.vector_store.search(question, n_results=5)
            
            # Prepare context from search results
            context_parts = []
            sources = []
            
            for result in search_results:
                context_parts.append(result['document'])
                sources.append({
                    'filename': result['metadata'].get('filename', 'Unknown'),
                    'candidate_name': result['metadata'].get('candidate_name', 'Unknown'),
                    'similarity': result['similarity'],
                    'content': result['document'][:500] + "..." if len(result['document']) > 500 else result['document']
                })
            
            context = "\n\n".join(context_parts)
            
            # Create the full prompt
            full_prompt = self.prompt_template.format(
                context=context,
                question=question
            )
            
            # Generate response using Gemini
            response = self.model.generate_content(full_prompt)
            
            return {
                "question": question,
                "answer": response.text,
                "sources": sources,
                "context_used": len(context_parts)
            }
            
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to Gemini API"""
        try:
            test_response = self.model.generate_content("Hello, this is a test.")
            return True
        except Exception as e:
            print(f"Gemini API connection test failed: {str(e)}")
            return False
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """Process multiple queries in batch"""
        results = []
        for question in questions:
            try:
                result = self.query(question)
                results.append(result)
            except Exception as e:
                results.append({
                    "question": question,
                    "error": str(e)
                })
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "model_name": "gemini-pro",
            "vector_store_stats": vector_stats,
            "api_configured": bool(self.api_key)
        }
    
    def add_cv_to_database(self, cv_data: CVData) -> str:
        """Add a CV to the vector database"""
        return self.vector_store.add_cv(cv_data)
    
    def clear_database(self) -> bool:
        """Clear all CVs from the database"""
        return self.vector_store.clear_all()