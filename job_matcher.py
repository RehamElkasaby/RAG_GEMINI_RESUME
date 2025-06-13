import re
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from cv_schema import CVData, MatchResult

class JobMatcher:
    """Job description matching system with explainable AI results"""
    
    def __init__(self):
        # Initialize embedding model for semantic similarity
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Common skill categories and their weights
        self.skill_categories = {
            'programming_languages': ['python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go'],
            'frameworks': ['react', 'angular', 'vue', 'django', 'flask', 'spring', 'node.js'],
            'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch'],
            'cloud_technologies': ['aws', 'azure', 'gcp', 'docker', 'kubernetes'],
            'data_science': ['machine learning', 'deep learning', 'data science', 'tensorflow', 'pytorch'],
            'tools': ['git', 'jenkins', 'jira', 'confluence']
        }
        
        # Weights for different matching criteria
        self.weights = {
            'skills': 0.5,
            'experience': 0.3,
            'education': 0.2
        }
    
    def find_matches(
        self, 
        job_description: str, 
        cv_data_list: List[Dict[str, Any]], 
        top_k: int = 5,
        include_explanations: bool = True
    ) -> List[MatchResult]:
        """Find top matching candidates for a job description"""
        
        if not cv_data_list:
            return []
        
        # Extract job requirements
        job_requirements = self._extract_job_requirements(job_description)
        
        # Calculate matches for each CV
        match_results = []
        for cv_data_dict in cv_data_list:
            try:
                # Convert dict to CVData object
                cv_data = CVData(**cv_data_dict)
                
                # Calculate matching scores
                scores = self._calculate_matching_scores(cv_data, job_requirements, job_description)
                
                # Create match result
                match_result = MatchResult(
                    cv_filename=cv_data.filename,
                    candidate_name=cv_data.personal_info.name or "Unknown",
                    overall_score=scores['overall'],
                    skill_match_score=scores['skills'],
                    experience_match_score=scores['experience'],
                    education_match_score=scores['education'],
                    explanation=self._generate_explanation(cv_data, job_requirements, scores) if include_explanations else "",
                    matched_skills=scores['matched_skills'],
                    relevant_experience=scores['relevant_experience'],
                    cv_data=cv_data_dict
                )
                
                match_results.append(match_result)
                
            except Exception as e:
                print(f"Error processing CV {cv_data_dict.get('filename', 'Unknown')}: {str(e)}")
                continue
        
        # Sort by overall score and return top K
        match_results.sort(key=lambda x: x.overall_score, reverse=True)
        return match_results[:top_k]
    
    def _extract_job_requirements(self, job_description: str) -> Dict[str, Any]:
        """Extract requirements from job description"""
        job_desc_lower = job_description.lower()
        
        # Extract required skills
        required_skills = set()
        for category, skills in self.skill_categories.items():
            for skill in skills:
                if skill in job_desc_lower:
                    required_skills.add(skill)
        
        # Extract experience requirements
        experience_patterns = [
            r'(\d+)\+?\s*years?\s*(?:of\s*)?experience',
            r'(\d+)\+?\s*years?\s*(?:of\s*)?(?:relevant\s*)?experience',
            r'minimum\s*(\d+)\s*years?'
        ]
        
        required_experience_years = 0
        for pattern in experience_patterns:
            match = re.search(pattern, job_desc_lower)
            if match:
                required_experience_years = max(required_experience_years, int(match.group(1)))
        
        # Extract education requirements
        education_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 'college']
        requires_degree = any(keyword in job_desc_lower for keyword in education_keywords)
        
        # Extract job level indicators
        senior_keywords = ['senior', 'lead', 'principal', 'architect', 'manager']
        is_senior_role = any(keyword in job_desc_lower for keyword in senior_keywords)
        
        return {
            'required_skills': list(required_skills),
            'required_experience_years': required_experience_years,
            'requires_degree': requires_degree,
            'is_senior_role': is_senior_role,
            'job_description_text': job_description
        }
    
    def _calculate_matching_scores(
        self, 
        cv_data: CVData, 
        job_requirements: Dict[str, Any],
        job_description: str
    ) -> Dict[str, Any]:
        """Calculate detailed matching scores"""
        
        # Skill matching
        skill_score, matched_skills = self._calculate_skill_match(cv_data, job_requirements)
        
        # Experience matching
        experience_score, relevant_experience = self._calculate_experience_match(cv_data, job_requirements, job_description)
        
        # Education matching
        education_score = self._calculate_education_match(cv_data, job_requirements)
        
        # Overall weighted score
        overall_score = (
            skill_score * self.weights['skills'] +
            experience_score * self.weights['experience'] +
            education_score * self.weights['education']
        )
        
        return {
            'overall': overall_score,
            'skills': skill_score,
            'experience': experience_score,
            'education': education_score,
            'matched_skills': matched_skills,
            'relevant_experience': relevant_experience
        }
    
    def _calculate_skill_match(self, cv_data: CVData, job_requirements: Dict[str, Any]) -> tuple:
        """Calculate skill matching score"""
        required_skills = job_requirements['required_skills']
        candidate_skills = [skill.name.lower() for skill in cv_data.skills]
        
        if not required_skills:
            return 0.5, []  # Neutral score if no specific skills required
        
        # Direct skill matches
        matched_skills = []
        for req_skill in required_skills:
            if req_skill in candidate_skills:
                matched_skills.append(req_skill)
        
        # Semantic similarity for skills not directly matched
        if len(matched_skills) < len(required_skills):
            # Use embeddings for semantic matching
            for req_skill in required_skills:
                if req_skill not in matched_skills:
                    req_embedding = self.embedding_model.encode([req_skill])
                    for candidate_skill in candidate_skills:
                        if candidate_skill not in matched_skills:
                            candidate_embedding = self.embedding_model.encode([candidate_skill])
                            similarity = cosine_similarity(req_embedding, candidate_embedding)[0][0]
                            if similarity > 0.7:  # Threshold for semantic similarity
                                matched_skills.append(candidate_skill)
                                break
        
        # Calculate score
        if not required_skills:
            skill_score = 0.5
        else:
            skill_score = len(matched_skills) / len(required_skills)
            # Bonus for having more skills than required
            if len(candidate_skills) > len(required_skills):
                skill_score = min(1.0, skill_score * 1.1)
        
        return skill_score, matched_skills
    
    def _calculate_experience_match(self, cv_data: CVData, job_requirements: Dict[str, Any], job_description: str) -> tuple:
        """Calculate experience matching score"""
        required_years = job_requirements['required_experience_years']
        candidate_years = cv_data.get_total_experience_years()
        
        # Find relevant experience based on job description
        relevant_experience = []
        job_desc_lower = job_description.lower()
        
        for exp in cv_data.experience:
            # Check if experience is relevant based on job title or description
            exp_text = f"{exp.job_title} {exp.description}".lower()
            
            # Simple keyword matching for relevance
            relevance_score = 0
            for word in job_desc_lower.split():
                if len(word) > 3 and word in exp_text:
                    relevance_score += 1
            
            if relevance_score > 0:
                relevant_experience.append(f"{exp.job_title} at {exp.company}")
        
        # Calculate experience score
        if required_years == 0:
            experience_score = 0.7 if candidate_years > 0 else 0.3
        else:
            if candidate_years >= required_years:
                experience_score = min(1.0, candidate_years / required_years)
            else:
                experience_score = candidate_years / required_years
        
        # Adjust for senior roles
        if job_requirements['is_senior_role'] and candidate_years < 3:
            experience_score *= 0.7
        
        return experience_score, relevant_experience
    
    def _calculate_education_match(self, cv_data: CVData, job_requirements: Dict[str, Any]) -> float:
        """Calculate education matching score"""
        requires_degree = job_requirements['requires_degree']
        has_degree = len(cv_data.education) > 0
        
        if not requires_degree:
            return 0.7  # Neutral score when no degree required
        
        if has_degree:
            # Check for advanced degrees
            degree_levels = []
            for edu in cv_data.education:
                degree_lower = edu.degree.lower()
                if 'phd' in degree_lower or 'doctorate' in degree_lower:
                    degree_levels.append(4)
                elif 'master' in degree_lower:
                    degree_levels.append(3)
                elif 'bachelor' in degree_lower:
                    degree_levels.append(2)
                else:
                    degree_levels.append(1)
            
            # Score based on highest degree
            max_degree_level = max(degree_levels) if degree_levels else 1
            education_score = min(1.0, max_degree_level / 2)
            
            return education_score
        else:
            return 0.3  # Low score if degree required but not present
    
    def _generate_explanation(self, cv_data: CVData, job_requirements: Dict[str, Any], scores: Dict[str, Any]) -> str:
        """Generate explanation for the match"""
        explanation_parts = []
        
        # Overall assessment
        overall_score = scores['overall']
        if overall_score >= 0.8:
            explanation_parts.append("🟢 Excellent match for this position.")
        elif overall_score >= 0.6:
            explanation_parts.append("🟡 Good match with some areas for consideration.")
        else:
            explanation_parts.append("🔴 Limited match for this position.")
        
        # Skills assessment
        matched_skills = scores['matched_skills']
        required_skills = job_requirements['required_skills']
        
        if matched_skills:
            explanation_parts.append(f"✅ Matching skills: {', '.join(matched_skills)}")
        
        if len(matched_skills) < len(required_skills):
            missing_skills = [skill for skill in required_skills if skill not in matched_skills]
            explanation_parts.append(f"❌ Missing skills: {', '.join(missing_skills)}")
        
        # Experience assessment
        candidate_years = cv_data.get_total_experience_years()
        required_years = job_requirements['required_experience_years']
        
        if required_years > 0:
            if candidate_years >= required_years:
                explanation_parts.append(f"✅ Has {candidate_years:.1f} years of experience (required: {required_years})")
            else:
                explanation_parts.append(f"⚠️ Has {candidate_years:.1f} years of experience (required: {required_years})")
        
        # Relevant experience
        relevant_exp = scores['relevant_experience']
        if relevant_exp:
            explanation_parts.append(f"🎯 Relevant experience: {', '.join(relevant_exp[:2])}")
        
        # Education assessment
        if job_requirements['requires_degree']:
            if cv_data.education:
                degrees = [edu.degree for edu in cv_data.education]
                explanation_parts.append(f"🎓 Education: {', '.join(degrees)}")
            else:
                explanation_parts.append("⚠️ No formal education information found")
        
        return " | ".join(explanation_parts)
    
    def bulk_match(self, job_descriptions: List[str], cv_data_list: List[Dict[str, Any]]) -> Dict[str, List[MatchResult]]:
        """Match multiple job descriptions against CV database"""
        results = {}
        
        for i, job_desc in enumerate(job_descriptions):
            job_key = f"Job_{i+1}"
            results[job_key] = self.find_matches(job_desc, cv_data_list)
        
        return results
    
    def export_matches_to_json(self, matches: List[MatchResult]) -> str:
        """Export match results to JSON format"""
        import json
        
        export_data = []
        for match in matches:
            export_item = {
                "candidate_name": match.candidate_name,
                "filename": match.cv_filename,
                "overall_score": round(match.overall_score, 3),
                "skill_match_score": round(match.skill_match_score, 3),
                "experience_match_score": round(match.experience_match_score, 3),
                "education_match_score": round(match.education_match_score, 3),
                "matched_skills": match.matched_skills,
                "relevant_experience": match.relevant_experience,
                "explanation": match.explanation
            }
            export_data.append(export_item)
        
        return json.dumps(export_data, indent=2)
