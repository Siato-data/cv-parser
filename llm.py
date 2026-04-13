import json
import time
import logging
from typing import Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

MODEL = "gpt-5.4-nano"
MAX_RETRIES = 3

SYSTEM_PROMPT = """Tu es un parser ATS expert. Tu extrais les informations d'un CV
et retournes UNIQUEMENT un JSON valide, sans markdown, sans commentaire.
Si une information est absente, utilise null. Ne l'invente jamais.

RÈGLE MULTILINGUE : Les clés JSON doivent toujours rester en anglais.
Les valeurs doivent être dans la langue originale du CV.
Par exemple, si le CV est en anglais, les valeurs sont en anglais.
Si le CV est en français, les valeurs sont en français.
Si le CV mélange plusieurs langues, garde la langue dominante pour les valeurs."""

USER_PROMPT = """Extrais les informations de ce CV.{job_section}

CV :
{resume_text}

JSON attendu :
{{
  "full_name": "string",
  "professional_title": "string",
  "seniority": "Junior|Mid|Senior|Lead|Executive",
  "years_of_experience": 0,
  "contact": {{
    "email": null,
    "phone": null,
    "linkedin": null,
    "location": null
  }},
  "summary": "résumé professionnel 2-3 phrases",
  "work_experience": [
    {{
      "title": "string",
      "company": "string",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM ou null si poste actuel",
      "achievements": ["réalisations concrètes, chiffrées si possible"],
      "technologies": ["outils, logiciels, langages utilisés"]
    }}
  ],
  "education": [
    {{
      "degree": "string",
      "field": "string",
      "institution": "string",
      "year": "YYYY"
    }}
  ],
  "skills": {{
    "technical": ["liste"],
    "soft": ["liste"],
    "languages": ["ex: Français (natif)", "Anglais (courant)"]
  }},
  "certifications": ["liste"],
  "hr_notes": {{
    "strengths": ["top 3 points forts"],
    "potential_roles": ["rôles suggérés selon profil"],
    "gaps": ["points de développement"]
  }},
  "job_match": {job_match_field}
}}"""

JOB_SECTION = """

Fiche de poste à matcher :
---
{job_description}
---
Évalue le matching dans job_match."""

JOB_MATCH_WITH_JD = """{
    "score": 0,
    "matched_skills": ["compétences présentes et demandées"],
    "missing_skills": ["compétences requises absentes"],
    "verdict": "une phrase sur l'adéquation globale"
  }"""

JOB_MATCH_NULL = "null"


def parse(client: OpenAI, resume_text: str, job_description: Optional[str] = None) -> dict:
    job_section = JOB_SECTION.format(job_description=job_description) if job_description else ""
    job_match_field = JOB_MATCH_WITH_JD if job_description else JOB_MATCH_NULL

    prompt = USER_PROMPT.format(
        resume_text=resume_text,
        job_section=job_section,
        job_match_field=job_match_field,
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            result = json.loads(response.choices[0].message.content.strip())
            result["_tokens"] = response.usage.total_tokens
            return result
        except Exception as e:
            last_error = e
            logger.warning(f"Tentative {attempt + 1}/{MAX_RETRIES} : {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Parsing échoué après {MAX_RETRIES} tentatives : {last_error}")
