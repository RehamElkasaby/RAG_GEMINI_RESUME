from typing import List, Optional, Dict, Any
from datetime import datetime
import json

class PersonalInfo:
    """Personal information schema"""
    def __init__(self, name: str = "", email: str = "", phone: str = "", 
                 location: str = "", linkedin: str = "", github: str = ""):
        self.name = name
        self.email = email
        self.phone = phone
        self.location = location
        self.linkedin = linkedin
        self.github = github
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "location": self.location,
            "linkedin": self.linkedin,
            "github": self.github
        }

class Skill:
    """Skill schema"""
    def __init__(self, name: str, category: str = "Technology", 
                 proficiency: str = "Intermediate", years_of_experience: Optional[int] = None):
        self.name = name
        self.category = category
        self.proficiency = proficiency
        self.years_of_experience = years_of_experience
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "proficiency": self.proficiency,
            "years_of_experience": self.years_of_experience
        }

class Experience:
    """Work experience schema"""
    def __init__(self, job_title: str, company: str, start_date: str = "", 
                 end_date: str = "", description: str = "", location: str = ""):
        self.job_title = job_title
        self.company = company
        self.start_date = start_date
        self.end_date = end_date
        self.description = description
        self.location = location
    
    def get_duration_years(self) -> float:
        """Calculate duration in years"""
        try:
            start_year = int(self.start_date.split('/')[0] if '/' in self.start_date else self.start_date)
            if self.end_date.lower() in ['present', 'current']:
                end_year = datetime.now().year
            else:
                end_year = int(self.end_date.split('/')[0] if '/' in self.end_date else self.end_date)
            return max(0, end_year - start_year)
        except:
            return 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_title": self.job_title,
            "company": self.company,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "description": self.description,
            "location": self.location
        }

class Education:
    """Education schema"""
    def __init__(self, degree: str, institution: str, field_of_study: str = "", 
                 graduation_year: str = "", gpa: str = "", honors: str = ""):
        self.degree = degree
        self.institution = institution
        self.field_of_study = field_of_study
        self.graduation_year = graduation_year
        self.gpa = gpa
        self.honors = honors
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "degree": self.degree,
            "institution": self.institution,
            "field_of_study": self.field_of_study,
            "graduation_year": self.graduation_year,
            "gpa": self.gpa,
            "honors": self.honors
        }

class Project:
    """Project schema"""
    def __init__(self, title: str, description: str = "", technologies: List[str] = None, 
                 url: str = "", start_date: str = "", end_date: str = ""):
        self.title = title
        self.description = description
        self.technologies = technologies or []
        self.url = url
        self.start_date = start_date
        self.end_date = end_date
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "technologies": self.technologies,
            "url": self.url,
            "start_date": self.start_date,
            "end_date": self.end_date
        }

class Certification:
    """Certification schema"""
    def __init__(self, name: str, issuing_organization: str, issue_date: str = "", 
                 expiry_date: str = "", credential_id: str = ""):
        self.name = name
        self.issuing_organization = issuing_organization
        self.issue_date = issue_date
        self.expiry_date = expiry_date
        self.credential_id = credential_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "issuing_organization": self.issuing_organization,
            "issue_date": self.issue_date,
            "expiry_date": self.expiry_date,
            "credential_id": self.credential_id
        }

class CVData:
    """Complete CV data schema"""
    def __init__(self, filename: str, raw_text: str, personal_info: PersonalInfo = None,
                 skills: List[Skill] = None, experience: List[Experience] = None, 
                 education: List[Education] = None, projects: List[Project] = None,
                 certifications: List[Certification] = None, languages: List[str] = None,
                 parsed_date: str = ""):
        self.filename = filename
        self.raw_text = raw_text
        self.personal_info = personal_info or PersonalInfo()
        self.skills = skills or []
        self.experience = experience or []
        self.education = education or []
        self.projects = projects or []
        self.certifications = certifications or []
        self.languages = languages or []
        self.parsed_date = parsed_date or datetime.now().isoformat()
    
    def get_total_experience_years(self) -> float:
        """Calculate total years of work experience"""
        return sum(exp.get_duration_years() for exp in self.experience)
    
    def get_skills_by_category(self) -> Dict[str, List[str]]:
        """Group skills by category"""
        skills_by_category = {}
        for skill in self.skills:
            if skill.category not in skills_by_category:
                skills_by_category[skill.category] = []
            skills_by_category[skill.category].append(skill.name)
        return skills_by_category
    
    def to_searchable_text(self) -> str:
        """Convert CV data to searchable text for vector embedding"""
        searchable_parts = []
        
        # Personal info
        if self.personal_info.name:
            searchable_parts.append(f"Name: {self.personal_info.name}")
        if self.personal_info.location:
            searchable_parts.append(f"Location: {self.personal_info.location}")
        
        # Skills
        if self.skills:
            skill_names = [skill.name for skill in self.skills]
            searchable_parts.append(f"Skills: {', '.join(skill_names)}")
        
        # Experience
        for exp in self.experience:
            exp_text = f"Experience: {exp.job_title} at {exp.company}"
            if exp.description:
                exp_text += f" - {exp.description}"
            searchable_parts.append(exp_text)
        
        # Education
        for edu in self.education:
            edu_text = f"Education: {edu.degree}"
            if edu.institution:
                edu_text += f" from {edu.institution}"
            if edu.field_of_study:
                edu_text += f" in {edu.field_of_study}"
            searchable_parts.append(edu_text)
        
        return "\n".join(searchable_parts)
    
    def dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "filename": self.filename,
            "raw_text": self.raw_text,
            "personal_info": self.personal_info.to_dict(),
            "skills": [skill.to_dict() for skill in self.skills],
            "experience": [exp.to_dict() for exp in self.experience],
            "education": [edu.to_dict() for edu in self.education],
            "projects": [proj.to_dict() for proj in self.projects],
            "certifications": [cert.to_dict() for cert in self.certifications],
            "languages": self.languages,
            "parsed_date": self.parsed_date
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CVData":
        """Create CVData from dictionary"""
        personal_info = PersonalInfo(**data.get("personal_info", {}))
        
        skills = [Skill(**skill_data) for skill_data in data.get("skills", [])]
        experience = [Experience(**exp_data) for exp_data in data.get("experience", [])]
        education = [Education(**edu_data) for edu_data in data.get("education", [])]
        projects = [Project(**proj_data) for proj_data in data.get("projects", [])]
        certifications = [Certification(**cert_data) for cert_data in data.get("certifications", [])]
        
        return cls(
            filename=data.get("filename", ""),
            raw_text=data.get("raw_text", ""),
            personal_info=personal_info,
            skills=skills,
            experience=experience,
            education=education,
            projects=projects,
            certifications=certifications,
            languages=data.get("languages", []),
            parsed_date=data.get("parsed_date", "")
        )

class MatchResult:
    """Job match result schema"""
    def __init__(self, cv_filename: str, candidate_name: str, overall_score: float = 0.0,
                 skill_match_score: float = 0.0, experience_match_score: float = 0.0,
                 education_match_score: float = 0.0, explanation: str = "",
                 matched_skills: List[str] = None, relevant_experience: List[str] = None,
                 cv_data: Dict[str, Any] = None):
        self.cv_filename = cv_filename
        self.candidate_name = candidate_name
        self.overall_score = overall_score
        self.skill_match_score = skill_match_score
        self.experience_match_score = experience_match_score
        self.education_match_score = education_match_score
        self.explanation = explanation
        self.matched_skills = matched_skills or []
        self.relevant_experience = relevant_experience or []
        self.cv_data = cv_data or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "cv_filename": self.cv_filename,
            "candidate_name": self.candidate_name,
            "overall_score": self.overall_score,
            "skill_match_score": self.skill_match_score,
            "experience_match_score": self.experience_match_score,
            "education_match_score": self.education_match_score,
            "explanation": self.explanation,
            "matched_skills": self.matched_skills,
            "relevant_experience": self.relevant_experience,
            "cv_data": self.cv_data
        }