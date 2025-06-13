import streamlit as st
import json
import os
from typing import List, Dict, Any
import pandas as pd

from cv_parser_simple import SimpleTextCVParser
from cv_schema_simple import CVData
from vector_store_simple import SimpleVectorStore
from job_matcher_simple import SimpleJobMatcher
from utils_simple import init_session_state, display_cv_summary, display_match_results

# Page configuration
st.set_page_config(
    page_title="Smart Recruiter Assistant",
    page_icon="👔",
    layout="wide"
)

# Initialize session state
init_session_state()

def main():
    st.title("👔 Smart Recruiter Assistant")
    st.markdown("**RAG-based CV Query and Job Matching System**")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a function:",
        ["CV Upload & Management", "CV Chatbot Q&A", "Job Description Matching"]
    )
    
    if page == "CV Upload & Management":
        cv_upload_page()
    elif page == "CV Chatbot Q&A":
        chatbot_page()
    elif page == "Job Description Matching":
        job_matching_page()

def cv_upload_page():
    st.header("📄 CV Upload & Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload CVs (PDF, DOCX, or TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload multiple CV files to build your candidate database"
    )
    
    # Text input option for manual CV entry
    st.subheader("Or paste CV content directly:")
    cv_text_input = st.text_area(
        "Paste CV content here:",
        height=200,
        placeholder="Copy and paste the CV content here..."
    )
    
    cv_filename_input = st.text_input(
        "CV filename:",
        placeholder="e.g., john_doe_cv.txt"
    )
    
    if cv_text_input and cv_filename_input:
        if st.button("Process Pasted CV", type="secondary"):
            process_text_cv(cv_text_input, cv_filename_input)
    
    if uploaded_files:
        process_button = st.button("Process CVs", type="primary")
        
        if process_button:
            process_uploaded_cvs(uploaded_files)
    
    # Display existing CVs
    if st.session_state.cv_data:
        st.subheader("📊 Processed CVs")
        display_processed_cvs()

def process_text_cv(text_content, filename):
    """Process CV text content directly"""
    try:
        parser = SimpleTextCVParser()
        vector_store = SimpleVectorStore()
        
        # Parse CV text
        cv_data = parser.parse_cv_text(text_content, filename)
        
        if cv_data:
            # Store in session state
            st.session_state.cv_data[filename] = cv_data.dict()
            
            # Add to vector store
            vector_store.add_cv(cv_data)
            
            st.success(f"✅ Successfully processed {filename}")
            st.session_state.vector_store_initialized = True
        else:
            st.error(f"❌ Failed to process {filename}")
            
    except Exception as e:
        st.error(f"❌ Error processing {filename}: {str(e)}")

def process_uploaded_cvs(uploaded_files):
    """Process uploaded CV files and store them in the vector database"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    parser = SimpleTextCVParser()
    vector_store = SimpleVectorStore()
    
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        temp_path = None
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Get file extension
            file_ext = os.path.splitext(uploaded_file.name)[1].lower()
            
            if file_ext in ['.pdf', '.docx', '.txt']:
                # Save uploaded file temporarily for all supported formats
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Parse CV file
                cv_data = parser.parse_cv_file(temp_path, uploaded_file.name)
            else:
                st.warning(f"⚠️ {uploaded_file.name}: Unsupported file format. Please use PDF, DOCX, or TXT files.")
                cv_data = None
            
            if cv_data:
                # Store in session state
                st.session_state.cv_data[uploaded_file.name] = cv_data.dict()
                
                # Add to vector store
                vector_store.add_cv(cv_data)
                
                st.success(f"✅ Successfully processed {uploaded_file.name}")
            else:
                st.error(f"❌ Failed to process {uploaded_file.name}")
            
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
                
        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text("Processing complete!")
    st.session_state.vector_store_initialized = True

def display_processed_cvs():
    """Display summary of processed CVs"""
    cv_summaries = []
    
    for filename, cv_data in st.session_state.cv_data.items():
        summary = {
            "Filename": filename,
            "Name": cv_data.get("personal_info", {}).get("name", "N/A"),
            "Email": cv_data.get("personal_info", {}).get("email", "N/A"),
            "Skills Count": len(cv_data.get("skills", [])),
            "Experience Years": len(cv_data.get("experience", [])),
            "Education Count": len(cv_data.get("education", []))
        }
        cv_summaries.append(summary)
    
    if cv_summaries:
        df = pd.DataFrame(cv_summaries)
        st.dataframe(df, use_container_width=True)
        
        # Export functionality
        if st.button("📥 Export CV Data as JSON"):
            json_data = json.dumps(st.session_state.cv_data, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="cv_database.json",
                mime="application/json"
            )

def chatbot_page():
    st.header("🤖 CV Chatbot Q&A")
    
    if not st.session_state.cv_data:
        st.warning("⚠️ Please upload and process CVs first in the 'CV Upload & Management' section.")
        return
    
    # Initialize simple search system
    if not hasattr(st.session_state, 'vector_store'):
        st.session_state.vector_store = SimpleVectorStore()
    
    # Chat interface
    st.subheader("💬 Ask questions about your CV database")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        - "Who has experience in machine learning?"
        - "Which candidates graduated from Stanford University?"
        - "Who has worked as a software engineer?"
        - "Find candidates with Python programming skills"
        - "Who has more than 5 years of experience?"
        """)
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="e.g., Who has good experience in Time Series?"
    )
    
    if query and st.button("🔍 Search", type="primary"):
        with st.spinner("Searching through CVs..."):
            try:
                # Simple text-based search through CV data
                search_results = []
                query_lower = query.lower()
                
                for filename, cv_data in st.session_state.cv_data.items():
                    # Create searchable text from CV data
                    cv_obj = CVData.from_dict(cv_data)
                    searchable_text = cv_obj.to_searchable_text().lower()
                    
                    # Simple keyword matching
                    relevance_score = 0
                    query_words = query_lower.split()
                    
                    for word in query_words:
                        if len(word) > 2 and word in searchable_text:
                            relevance_score += 1
                    
                    if relevance_score > 0:
                        search_results.append({
                            'filename': filename,
                            'candidate_name': cv_data.get('personal_info', {}).get('name', 'Unknown'),
                            'relevance_score': relevance_score,
                            'cv_data': cv_data
                        })
                
                # Sort by relevance
                search_results.sort(key=lambda x: x['relevance_score'], reverse=True)
                
                if search_results:
                    st.subheader("📋 Search Results")
                    
                    for i, result in enumerate(search_results[:5], 1):
                        with st.expander(f"#{i} {result['candidate_name']} - {result['filename']}"):
                            display_cv_summary(result['cv_data'])
                else:
                    st.info("No matching candidates found for your query.")
                            
            except Exception as e:
                st.error(f"❌ Error processing query: {str(e)}")

def job_matching_page():
    st.header("🎯 Job Description Matching")
    
    if not st.session_state.cv_data:
        st.warning("⚠️ Please upload and process CVs first in the 'CV Upload & Management' section.")
        return
    
    # Job description input
    st.subheader("📝 Enter Job Description")
    job_description = st.text_area(
        "Job Description:",
        height=200,
        placeholder="Paste the job description here..."
    )
    
    # Matching parameters
    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Number of top candidates to show:", 1, 10, 5)
    with col2:
        include_explanations = st.checkbox("Include detailed explanations", value=True)
    
    if job_description and st.button("🔍 Find Matching Candidates", type="primary"):
        with st.spinner("Analyzing candidates..."):
            try:
                # Initialize job matcher
                matcher = SimpleJobMatcher()
                
                # Get matches
                matches = matcher.find_matches(
                    job_description, 
                    list(st.session_state.cv_data.values()),
                    top_k=top_k,
                    include_explanations=include_explanations
                )
                
                if matches:
                    st.subheader(f"🏆 Top {len(matches)} Matching Candidates")
                    display_match_results(matches)
                    
                    # Export functionality
                    if st.button("📥 Export Results as JSON"):
                        json_results = matcher.export_matches_to_json(matches)
                        st.download_button(
                            label="Download JSON",
                            data=json_results,
                            file_name="job_match_results.json",
                            mime="application/json"
                        )
                else:
                    st.info("No matching candidates found.")
                    
            except Exception as e:
                st.error(f"❌ Error during matching: {str(e)}")

if __name__ == "__main__":
    main()
