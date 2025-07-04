import os
import json
from typing import Optional, Dict, Any, List
import re
from datetime import datetime

from cv_schema import CVData, PersonalInfo, Experience, Education, Skill

# Optional imports for PDF and DOCX support
try:
    import fitz  # PyMuPDF
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

class CVParser:
    """Parser for extracting structured data from CV files"""
    
    def __init__(self):
        self.skill_keywords = [
            # Programming languages
            'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go', 'rust',
            'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css',
            # Frameworks and libraries
            'react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js', 'express',
            'tensorflow', 'pytorch', 'scikit-learn', 'pandas', 'numpy',
            # Technologies
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'jenkins', 'git', 'linux',
            'machine learning', 'deep learning', 'artificial intelligence', 'data science',
            'big data', 'blockchain', 'cloud computing', 'devops', 'agile', 'scrum',
            # Databases
            'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle',
            # Tools
            'jira', 'confluence', 'slack', 'trello', 'figma', 'sketch'
        ]
    
    def parse_cv(self, file_path: str, filename: str) -> Optional[CVData]:
        """Parse a CV file and return structured data"""
        try:
            # Extract text based on file extension
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext == '.pdf':
                text = self._extract_text_from_pdf(file_path)
            elif file_ext == '.docx':
                text = self._extract_text_from_docx(file_path)
            elif file_ext == '.txt':
                text = self._extract_text_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            if not text.strip():
                raise ValueError("No text content found in the file")
            
            # Parse structured data from text
            cv_data = self._parse_text_to_structured_data(text, filename)
            return cv_data
            
        except Exception as e:
            print(f"Error parsing CV {filename}: {str(e)}")
            return None
    
    def _extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        if not PDF_SUPPORT:
            raise ValueError("PDF support not available. Please install PyMuPDF: pip install PyMuPDF")
        
        text = ""
        try:
            doc = fitz.open(file_path)
            for page in doc:
                text += page.get_text()
            doc.close()
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
        return text
    
    def _extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        if not DOCX_SUPPORT:
            raise ValueError("DOCX support not available. Please install python-docx: pip install python-docx")
        
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {str(e)}")
        return text
    
    def _extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            raise ValueError(f"Failed to extract text from TXT: {str(e)}")
        return text
    
    def _parse_text_to_structured_data(self, text: str, filename: str) -> CVData:
        """Parse raw text into structured CV data"""
        
        # Extract personal information
        personal_info = self._extract_personal_info(text)
        
        # Extract skills
        skills = self._extract_skills(text)
        
        # Extract experience
        experience = self._extract_experience(text)
        
        # Extract education
        education = self._extract_education(text)
        
        # Create CVData object
        cv_data = CVData(
            filename=filename,
            raw_text=text,
            personal_info=personal_info,
            skills=skills,
            experience=experience,
            education=education,
            parsed_date=datetime.now().isoformat()
        )
        
        return cv_data
    
    def _extract_personal_info(self, text: str) -> PersonalInfo:
        """Extract personal information from text"""
        
        # Extract name (usually at the beginning)
        name_match = re.search(r'^([A-Z][a-z]+ [A-Z][a-z]+)', text, re.MULTILINE)
        name = name_match.group(1) if name_match else ""
        
        # Extract email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        email = email_match.group(0) if email_match else ""
        
        # Extract phone
        phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
        phone = phone_match.group(0) if phone_match else ""
        
        # Extract location (look for city, state patterns)
        location_patterns = [
            r'([A-Z][a-z]+,\s*[A-Z]{2})',  # City, ST
            r'([A-Z][a-z]+,\s*[A-Z][a-z]+)',  # City, State
        ]
        location = ""
        for pattern in location_patterns:
            location_match = re.search(pattern, text)
            if location_match:
                location = location_match.group(1)
                break
        
        return PersonalInfo(
            name=name,
            email=email,
            phone=phone,
            location=location
        )
    
    def _extract_skills(self, text: str) -> List[Skill]:
        """Extract skills from text"""
        skills = []
        text_lower = text.lower()
        
        # Find skills section
        skills_section = ""
        skills_patterns = [
            r'skills?\s*:?\s*(.*?)(?=\n\s*\n|\nexperience|\neducation|$)',
            r'technical\s*skills?\s*:?\s*(.*?)(?=\n\s*\n|\nexperience|\neducation|$)',
            r'competencies\s*:?\s*(.*?)(?=\n\s*\n|\nexperience|\neducation|$)'
        ]
        
        for pattern in skills_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                skills_section = match.group(1)
                break
        
        # If no skills section found, search in entire text
        if not skills_section:
            skills_section = text_lower
        
        # Extract skills from keywords
        found_skills = set()
        for skill_keyword in self.skill_keywords:
            if skill_keyword in skills_section:
                found_skills.add(skill_keyword.title())
        
        # Convert to Skill objects
        for skill_name in found_skills:
            skills.append(Skill(
                name=skill_name,
                category=self._categorize_skill(skill_name),
                proficiency="Intermediate"  # Default proficiency
            ))
        
        return skills
    
    def _categorize_skill(self, skill_name: str) -> str:
        """Categorize a skill into a category"""
        skill_lower = skill_name.lower()
        
        programming_languages = ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby']
        frameworks = ['react', 'angular', 'vue', 'django', 'flask', 'spring']
        databases = ['mysql', 'postgresql', 'mongodb', 'redis']
        cloud_tech = ['aws', 'azure', 'gcp', 'docker', 'kubernetes']
        
        if skill_lower in programming_languages:
            return "Programming Language"
        elif skill_lower in frameworks:
            return "Framework"
        elif skill_lower in databases:
            return "Database"
        elif skill_lower in cloud_tech:
            return "Cloud Technology"
        else:
            return "Technology"
    
    def _extract_experience(self, text: str) -> List[Experience]:
        """Extract work experience from text"""
        experiences = []
        
        # Look for experience sections
        exp_patterns = [
            r'experience\s*:?\s*(.*?)(?=\neducation|\nskills?|\nprojects?|$)',
            r'work\s*experience\s*:?\s*(.*?)(?=\neducation|\nskills?|\nprojects?|$)',
            r'employment\s*:?\s*(.*?)(?=\neducation|\nskills?|\nprojects?|$)'
        ]
        
        experience_section = ""
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                experience_section = match.group(1)
                break
        
        if experience_section:
            # Split by job entries (look for dates or company names)
            job_entries = re.split(r'\n(?=\d{4}|\w+\s+\d{4})', experience_section)
            
            for entry in job_entries:
                if entry.strip():
                    exp = self._parse_experience_entry(entry.strip())
                    if exp:
                        experiences.append(exp)
        
        return experiences
    
    def _parse_experience_entry(self, entry: str) -> Optional[Experience]:
        """Parse a single experience entry"""
        try:
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            
            if len(lines) < 2:
                return None
            
            # Try to extract job title and company
            job_title = ""
            company = ""
            start_date = ""
            end_date = ""
            
            # Look for patterns like "Job Title at Company"
            title_company_match = re.search(r'(.+?)\s+at\s+(.+)', lines[0])
            if title_company_match:
                job_title = title_company_match.group(1).strip()
                company = title_company_match.group(2).strip()
            else:
                job_title = lines[0]
                if len(lines) > 1:
                    company = lines[1]
            
            # Look for dates
            date_pattern = r'(\d{4})\s*[-–]\s*(\d{4}|present|current)'
            date_match = re.search(date_pattern, entry, re.IGNORECASE)
            if date_match:
                start_date = date_match.group(1)
                end_date = date_match.group(2)
            
            # Extract description (remaining lines)
            description_lines = lines[2:] if len(lines) > 2 else []
            description = ' '.join(description_lines)
            
            return Experience(
                job_title=job_title,
                company=company,
                start_date=start_date,
                end_date=end_date,
                description=description
            )
            
        except Exception:
            return None
    
    def _extract_education(self, text: str) -> List[Education]:
        """Extract education information from text"""
        education_list = []
        
        # Look for education section
        edu_patterns = [
            r'education\s*:?\s*(.*?)(?=\nexperience|\nskills?|\nprojects?|$)',
            r'academic\s*background\s*:?\s*(.*?)(?=\nexperience|\nskills?|\nprojects?|$)'
        ]
        
        education_section = ""
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                education_section = match.group(1)
                break
        
        if education_section:
            # Split by education entries
            edu_entries = re.split(r'\n(?=\d{4}|\w+\s+\d{4})', education_section)
            
            for entry in edu_entries:
                if entry.strip():
                    edu = self._parse_education_entry(entry.strip())
                    if edu:
                        education_list.append(edu)
        
        return education_list
    
    def _parse_education_entry(self, entry: str) -> Optional[Education]:
        """Parse a single education entry"""
        try:
            lines = [line.strip() for line in entry.split('\n') if line.strip()]
            
            if not lines:
                return None
            
            degree = ""
            institution = ""
            graduation_year = ""
            
            # Look for degree and institution
            if len(lines) >= 1:
                degree = lines[0]
            
            if len(lines) >= 2:
                institution = lines[1]
            
            # Look for graduation year
            year_match = re.search(r'\b(19|20)\d{2}\b', entry)
            if year_match:
                graduation_year = year_match.group(0)
            
            return Education(
                degree=degree,
                institution=institution,
                graduation_year=graduation_year,
                field_of_study=""
            )
            
        except Exception:
            return None
