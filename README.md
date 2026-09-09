# AI Resume Analyzer

An AI-powered resume analysis tool that compares a candidate's resume with a job description and evaluates their overall fit.

## Features

- Upload resumes in PDF or DOCX format
- AI-powered resume analysis using Google Gemini
- Evaluate Required Qualifications, Preferred Qualifications, and Job Responsibilities
- Generate weighted match scores
- Identify missing qualifications
- Provide resume improvement recommendations

## Tech Stack

- Python
- Streamlit
- Google Gemini API
- PyPDF
- python-docx

## How It Works

1. Upload a resume.
2. Paste the job description.
3. Click **Analyze Resume**.
4. Gemini analyzes the resume against the job description.
5. The application displays category-level scores, an overall match score, and recommendations.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
