from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class PersonalInfo(BaseModel):
    """Personal information schema"""
    name: str = Field(default="", description="Full name of the candidate")
    email: str = Field(default="", description="Email address")
    phone: str = Field(default="", description="Phone number")
    location: str = Field(default="", description="Current location")
    linkedin: Optional[str] = Field(default="", description="LinkedIn profile URL")
    github: Optional[str] = Field(default="", description="GitHub profile URL")

class Skill(BaseModel):
    """Skill schema"""
    name: str = Field(description="Name of the skill")
    category: str = Field(description="Category of the skill (e.g., Programming Language, Framework)")
    proficiency: str = Field(default="Intermediate", description="Proficiency level (Beginner, Intermediate, Advanced, Expert)")
    years_of_experience: Optional[int] = Field(default=None, description="Years of experience with this skill")

class Experience(BaseModel):
    """Work experience schema"""
    job_title: str = Field(description="Job title/position")
    company: str = Field(description="Company name")
    start_date: str = Field(description="Start date (YYYY or MM/YYYY format)")
    end_date: str = Field(description="End date (YYYY or MM/YYYY format, or 'Present')")
    description: str = Field(default="", description="Job description and responsibilities")
    location: Optional[str] = Field(default="", description="Job location")
    
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

class Education(BaseModel):
    """Education schema"""
    degree: str = Field(description="Degree name (e.g., Bachelor's, Master's, PhD)")
    institution: str = Field(description="Name of the educational institution")
    field_of_study: str = Field(default="", description="Field of study or major")
    graduation_year: str = Field(description="Graduation year")
    gpa: Optional[str] = Field(default="", description="GPA if available")
    honors: Optional[str] = Field(default="", description="Academic honors or achievements")

class Project(BaseModel):
    """Project schema"""
    title: str = Field(description="Project title")
    description: str = Field(description="Project description")
    technologies: List[str] = Field(default_factory=list, description="Technologies used in the project")
    url: Optional[str] = Field(default="", description="Project URL if available")
    start_date: Optional[str] = Field(default="", description="Project start date")
    end_date: Optional[str] = Field(default="", description="Project end date")

class Certification(BaseModel):
    """Certification schema"""
    name: str = Field(description="Certification name")
    issuing_organization: str = Field(description="Organization that issued the certification")
    issue_date: Optional[str] = Field(default="", description="Date when certification was issued")
    expiry_date: Optional[str] = Field(default="", description="Certification expiry date")
    credential_id: Optional[str] = Field(default="", description="Certification credential ID")

class CVData(BaseModel):
    """Complete CV data schema"""
    filename: str = Field(description="Original filename of the CV")
    raw_text: str = Field(description="Raw extracted text from the CV")
    personal_info: PersonalInfo = Field(description="Personal information")
    skills: List[Skill] = Field(default_factory=list, description="List of skills")
    experience: List[Experience] = Field(default_factory=list, description="Work experience")
    education: List[Education] = Field(default_factory=list, description="Education background")
    projects: List[Project] = Field(default_factory=list, description="Projects")
    certifications: List[Certification] = Field(default_factory=list, description="Certifications")
    languages: List[str] = Field(default_factory=list, description="Languages spoken")
    parsed_date: str = Field(description="Date when CV was parsed")
    
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

class MatchResult(BaseModel):
    """Job match result schema"""
    cv_filename: str = Field(description="Filename of the matched CV")
    candidate_name: str = Field(description="Name of the candidate")
    overall_score: float = Field(description="Overall matching score (0-1)")
    skill_match_score: float = Field(description="Skill matching score (0-1)")
    experience_match_score: float = Field(description="Experience matching score (0-1)")
    education_match_score: float = Field(description="Education matching score (0-1)")
    explanation: str = Field(description="Explanation for the match")
    matched_skills: List[str] = Field(default_factory=list, description="Skills that matched the job")
    relevant_experience: List[str] = Field(default_factory=list, description="Relevant work experience")
    cv_data: Dict[str, Any] = Field(description="Complete CV data")
