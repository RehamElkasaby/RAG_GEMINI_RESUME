import os
from typing import List, Dict, Any, Optional
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.vectorstores.base import VectorStore as LangChainVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from vector_store import VectorStore

class ChromaVectorStoreWrapper(LangChainVectorStore):
    """Wrapper to make our ChromaDB vector store compatible with LangChain"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""
        results = self.vector_store.search(query, n_results=k)
        
        documents = []
        for result in results:
            doc = Document(
                page_content=result['document'],
                metadata={
                    'filename': result['metadata'].get('filename', 'Unknown'),
                    'candidate_name': result['metadata'].get('candidate_name', 'Unknown'),
                    'similarity': result['similarity']
                }
            )
            documents.append(doc)
        
        return documents
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> List[str]:
        """Add texts to the vector store (not implemented for this use case)"""
        raise NotImplementedError("Use add_cv method instead")

class RAGPipeline:
    """RAG pipeline for CV querying using LangChain and Ollama"""
    
    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.vector_store = VectorStore()
        self.vector_store_wrapper = ChromaVectorStoreWrapper(self.vector_store)
        
        # Initialize Ollama LLM
        self.llm = Ollama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        # Create prompt template
        self.prompt_template = self._create_prompt_template()
        
        # Initialize retrieval chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store_wrapper.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": self.prompt_template},
            return_source_documents=True
        )
    
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the prompt template for the RAG system"""
        template = """You are a helpful AI assistant specializing in recruitment and CV analysis. 
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
        
        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question"""
        try:
            # Execute the query
            result = self.qa_chain({"query": question})
            
            # Format the response
            response = {
                "question": question,
                "answer": result["result"],
                "sources": []
            }
            
            # Add source information
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source_info = {
                        "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                        "filename": doc.metadata.get("filename", "Unknown"),
                        "candidate_name": doc.metadata.get("candidate_name", "Unknown"),
                        "similarity": doc.metadata.get("similarity", 0.0)
                    }
                    response["sources"].append(source_info)
            
            return response
            
        except Exception as e:
            raise Exception(f"Error processing query: {str(e)}")
    
    def test_connection(self) -> bool:
        """Test connection to Ollama server"""
        try:
            # Try a simple query to test the connection
            test_response = self.llm("Hello, this is a test.")
            return True
        except Exception as e:
            print(f"Ollama connection test failed: {str(e)}")
            return False
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags")
            if response.status_code == 200:
                models = response.json()
                return [model["name"] for model in models.get("models", [])]
            return []
        except Exception as e:
            print(f"Failed to get available models: {str(e)}")
            return []
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different Ollama model"""
        try:
            self.model_name = model_name
            self.llm = Ollama(
                model=model_name,
                base_url="http://localhost:11434",
                temperature=0.7
            )
            
            # Recreate the QA chain with the new model
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store_wrapper.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": self.prompt_template},
                return_source_documents=True
            )
            
            return True
        except Exception as e:
            print(f"Failed to switch model: {str(e)}")
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
            "model_name": self.model_name,
            "vector_store_stats": vector_stats,
            "prompt_template_length": len(self.prompt_template.template)
        }
