import streamlit as st
import pandas as pd
from typing import List, Dict, Any
import json

from cv_schema_simple import MatchResult

def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'cv_data' not in st.session_state:
        st.session_state.cv_data = {}
    
    if 'vector_store_initialized' not in st.session_state:
        st.session_state.vector_store_initialized = False
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

def display_cv_summary(cv_data: Dict[str, Any]):
    """Display a summary of a CV"""
    personal_info = cv_data.get('personal_info', {})
    skills = cv_data.get('skills', [])
    experience = cv_data.get('experience', [])
    education = cv_data.get('education', [])
    
    st.subheader(f"📄 {personal_info.get('name', 'Unknown Name')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Contact Information:**")
        if personal_info.get('email'):
            st.write(f"📧 {personal_info['email']}")
        if personal_info.get('phone'):
            st.write(f"📱 {personal_info['phone']}")
        if personal_info.get('location'):
            st.write(f"📍 {personal_info['location']}")
    
    with col2:
        st.write("**Overview:**")
        st.write(f"🛠️ Skills: {len(skills)}")
        st.write(f"💼 Experience: {len(experience)} positions")
        st.write(f"🎓 Education: {len(education)} entries")
    
    # Skills section
    if skills:
        st.write("**Skills:**")
        skill_names = [skill['name'] for skill in skills]
        st.write(", ".join(skill_names))
    
    # Experience section
    if experience:
        st.write("**Recent Experience:**")
        for exp in experience[:2]:  # Show only first 2 experiences
            st.write(f"• {exp.get('job_title', 'Unknown')} at {exp.get('company', 'Unknown')}")

def display_match_results(matches: List[MatchResult]):
    """Display job matching results in a formatted way"""
    
    for i, match in enumerate(matches, 1):
        with st.container():
            # Header with rank and score
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.subheader(f"#{i} {match.candidate_name}")
                st.caption(f"📄 {match.cv_filename}")
            
            with col2:
                score_color = "green" if match.overall_score >= 0.7 else "orange" if match.overall_score >= 0.5 else "red"
                st.metric("Overall Score", f"{match.overall_score:.1%}", delta=None)
            
            with col3:
                st.write("**Breakdown:**")
                st.write(f"Skills: {match.skill_match_score:.1%}")
                st.write(f"Experience: {match.experience_match_score:.1%}")
                st.write(f"Education: {match.education_match_score:.1%}")
            
            # Detailed information
            if match.explanation:
                with st.expander("📋 Detailed Analysis"):
                    st.write(match.explanation)
            
            # Matched skills
            if match.matched_skills:
                st.write("**🎯 Matching Skills:**")
                skills_text = ", ".join(match.matched_skills)
                st.success(skills_text)
            
            # Relevant experience
            if match.relevant_experience:
                st.write("**💼 Relevant Experience:**")
                for exp in match.relevant_experience:
                    st.write(f"• {exp}")
            
            # Contact information
            personal_info = match.cv_data.get('personal_info', {})
            if personal_info.get('email') or personal_info.get('phone'):
                with st.expander("📞 Contact Information"):
                    if personal_info.get('email'):
                        st.write(f"📧 Email: {personal_info['email']}")
                    if personal_info.get('phone'):
                        st.write(f"📱 Phone: {personal_info['phone']}")
                    if personal_info.get('location'):
                        st.write(f"📍 Location: {personal_info['location']}")
            
            st.divider()

def create_cv_dataframe(cv_data_dict: Dict[str, Any]) -> pd.DataFrame:
    """Create a DataFrame from CV data for display"""
    rows = []
    
    for filename, cv_data in cv_data_dict.items():
        personal_info = cv_data.get('personal_info', {})
        skills = cv_data.get('skills', [])
        experience = cv_data.get('experience', [])
        education = cv_data.get('education', [])
        
        # Calculate total experience years
        total_years = 0
        for exp in experience:
            try:
                start_year = int(exp.get('start_date', '0').split('/')[0] if '/' in exp.get('start_date', '0') else exp.get('start_date', '0'))
                end_date = exp.get('end_date', 'present')
                if end_date.lower() in ['present', 'current']:
                    from datetime import datetime
                    end_year = datetime.now().year
                else:
                    end_year = int(end_date.split('/')[0] if '/' in end_date else end_date)
                total_years += max(0, end_year - start_year)
            except:
                pass
        
        row = {
            'Filename': filename,
            'Name': personal_info.get('name', 'N/A'),
            'Email': personal_info.get('email', 'N/A'),
            'Phone': personal_info.get('phone', 'N/A'),
            'Location': personal_info.get('location', 'N/A'),
            'Skills Count': len(skills),
            'Experience Years': total_years,
            'Education Count': len(education),
            'Top Skills': ', '.join([skill['name'] for skill in skills[:5]])
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

def validate_cv_data(cv_data: Dict[str, Any]) -> tuple:
    """Validate CV data structure and return validation results"""
    errors = []
    warnings = []
    
    # Check required fields
    if not cv_data.get('filename'):
        errors.append("Filename is missing")
    
    if not cv_data.get('raw_text'):
        errors.append("Raw text is missing")
    
    personal_info = cv_data.get('personal_info', {})
    if not personal_info.get('name'):
        warnings.append("Candidate name is missing")
    
    if not personal_info.get('email'):
        warnings.append("Email address is missing")
    
    # Check data completeness
    if not cv_data.get('skills'):
        warnings.append("No skills extracted")
    
    if not cv_data.get('experience'):
        warnings.append("No work experience found")
    
    if not cv_data.get('education'):
        warnings.append("No education information found")
    
    return errors, warnings

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def clean_text(text: str) -> str:
    """Clean and normalize text for better processing"""
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\-.,@()]', ' ', text)
    
    # Remove extra spaces
    text = text.strip()
    
    return text

def extract_keywords_from_text(text: str, min_length: int = 3) -> List[str]:
    """Extract keywords from text for search and matching"""
    import re
    from collections import Counter
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b[a-zA-Z]{' + str(min_length) + ',}\b', text.lower())
    
    # Common stop words to filter out
    stop_words = {
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'between', 'among', 'under', 'over', 'this', 'that',
        'these', 'those', 'his', 'her', 'its', 'their', 'our', 'your', 'has',
        'have', 'had', 'will', 'would', 'could', 'should', 'may', 'might',
        'can', 'must', 'shall', 'was', 'were', 'been', 'being', 'are', 'is'
    }
    
    # Filter out stop words
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequency and return most common
    word_counts = Counter(keywords)
    return [word for word, count in word_counts.most_common(50)]