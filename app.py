"""
AI Resume Analyzer
==================
A Streamlit application that analyzes a resume against a job description
using the Google Gemini API.

Architecture:
- PDF text extraction: pypdf
- AI analysis:         google-genai SDK  (model: gemini-3.6-flash)
- API key source:      Streamlit Secrets  →  st.secrets["GEMINI_API_KEY"]
- Deployment target:   Streamlit Community Cloud
"""

import json
import tempfile
import os
import time

import streamlit as st
from pypdf import PdfReader

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS  — clean, modern, professional
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global font & background ───────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Full-page soft gradient background ──────────────────── */
    .stApp {
        background:
            radial-gradient(
                circle at 52% 32%,
                rgba(174, 246, 220, 0.72) 0%,
                rgba(174, 246, 220, 0.45) 18%,
                rgba(174, 246, 220, 0.00) 42%
            ),
            radial-gradient(
                circle at 91% 55%,
                rgba(214, 202, 255, 0.72) 0%,
                rgba(214, 202, 255, 0.42) 24%,
                rgba(214, 202, 255, 0.00) 48%
            ),
            radial-gradient(
                circle at 67% 100%,
                rgba(226, 216, 255, 0.62) 0%,
                rgba(226, 216, 255, 0.00) 38%
            ),
            linear-gradient(
                135deg,
                #ffffff 0%,
                #fbfcfd 38%,
                #f7f9fc 70%,
                #f5f6fb 100%
            );
        background-attachment: fixed;
        min-height: 100vh;
    }

    /* Keep Streamlit's main containers transparent */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stMain"] {
        background: transparent !important;
    }

    /* Prevent the page from getting a second solid background */
    .main {
        background: transparent !important;
    }

    /* ── Header banner ───────────────────────────────────────── */
.hero-banner {
    background:
        linear-gradient(
            135deg,
            #171a35 0%,
            #182344 48%,
            #173f6c 100%
        );
    border-radius: 20px;
    padding: 54px 48px 50px 48px;
    margin: 14px 0 36px 0;
    text-align: center;
    box-shadow:
        0 18px 40px rgba(37, 50, 86, 0.22),
        0 4px 12px rgba(37, 50, 86, 0.10);
    border: 1px solid rgba(255, 255, 255, 0.10);
}

.hero-banner h1 {
    color: #ffffff;
    font-size: 2.7rem;
    font-weight: 700;
    margin: 0 0 14px 0;
    letter-spacing: -0.8px;
    line-height: 1.15;
}

.hero-banner p {
    color: #b9c4e5;
    font-size: 1.08rem;
    font-weight: 400;
    margin: 0;
    line-height: 1.65;
}

    /* ── Section cards ───────────────────────────────────────── */
    .section-card {
        background: rgba(255, 255, 255, 0.62);
backdrop-filter: blur(8px);
-webkit-backdrop-filter: blur(8px);
        border-radius: 12px;
        padding: 28px;
        border: 1px solid #e8ecf0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
        margin-bottom: 20px;
    }
    .section-label {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #6c7a93;
        margin-bottom: 8px;
    }

    /* ── Score badge ─────────────────────────────────────────── */
    .score-container {
        text-align: center;
        padding: 32px 20px;
    }
    .score-circle {
        display: inline-block;
        width: 140px;
        height: 140px;
        border-radius: 50%;
        line-height: 140px;
        font-size: 2.8rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 12px;
    }
    .score-high   { background: linear-gradient(135deg, #11998e, #38ef7d); }
    .score-medium { background: linear-gradient(135deg, #f09819, #edde5d); }
    .score-low    { background: linear-gradient(135deg, #e52d27, #b31217); }

    /* ── Skill pills ─────────────────────────────────────────── */
    .pill-container { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }

    .pill-match {
        background: #edfaf1;
        color: #1a7a4a;
        border: 1.5px solid #82e0aa;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.88rem;
        font-weight: 500;
    }
    .pill-missing {
        background: #fdf0f0;
        color: #a93226;
        border: 1.5px solid #f1948a;
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 0.88rem;
        font-weight: 500;
    }

    /* ── Recommendation box ──────────────────────────────────── */
    .recommendation-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin-top: 10px;
        color: #2c3e6b;
        font-size: 0.93rem;
        line-height: 1.6;
    }

    /* ── Step indicator ──────────────────────────────────────── */
    .step-badge {
        background: #4361ee;
        color: white;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 10px;
        vertical-align: middle;
    }

    /* ── Analyze Resume button ───────────────────────────────── */

.stButton > button {
    width: 220px;
    height: 52px;
    border-radius: 10px;
    border: 1px solid rgba(67, 97, 238, 0.15);
    background: linear-gradient(
        135deg,
        #4361ee 0%,
        #4d63e8 55%,
        #5368e8 100%
    );
    color: #ffffff;
    font-size: 1.02rem;
    font-weight: 600;
    letter-spacing: 0.1px;
    box-shadow: 0 8px 18px rgba(67, 97, 238, 0.20);
    transition:
        transform 0.15s ease,
        box-shadow 0.15s ease,
        background 0.15s ease;
}

.stButton > button:hover {
    background: linear-gradient(
        135deg,
        #3f5ce8 0%,
        #4a60e2 55%,
        #5065e5 100%
    );
    color: #ffffff;
    border-color: rgba(67, 97, 238, 0.25);
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(67, 97, 238, 0.27);
}

.stButton > button:active {
    transform: translateY(0);
    box-shadow: 0 5px 12px rgba(67, 97, 238, 0.20);
}

/* ── Matching Resume / Job Description input boxes ───────── */

/* Resume upload box */
[data-testid="stFileUploaderDropzone"] {
    height: 220px !important;
    min-height: 220px !important;
    box-sizing: border-box !important;

    background: #ffffff !important;
    border: 2px dashed #8b7cf6 !important;
    border-radius: 12px !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 0 24px !important;
}

/* Job description box */
[data-testid="stTextArea"] textarea {
    height: 220px !important;
    min-height: 220px !important;
    box-sizing: border-box !important;

    background: #ffffff !important;
    border: 2px dashed #8b7cf6 !important;
    border-radius: 12px !important;

    padding: 18px !important;
    resize: vertical !important;
}

/* Remove unnecessary uploader wrapper spacing */
[data-testid="stFileUploader"] {
    margin: 0 !important;
}

    /* ── Divider ─────────────────────────────────────────────── */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, #4361ee, transparent);
        border: none;
        margin: 24px 0;
    }

    /* ── Hide Streamlit default branding ─────────────────────── */
    #MainMenu, footer { visibility: hidden; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONSTANTS  — easy to change later
# ─────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-3.6-flash"   # ← change this to update the model
MAX_PDF_SIZE_MB = 10                   # maximum allowed upload size


# ─────────────────────────────────────────────────────────────
# HELPER: load the Gemini client
# ─────────────────────────────────────────────────────────────
def get_gemini_client():
    """
    Return a configured google-genai Client.

    API key priority:
      1. st.secrets["GEMINI_API_KEY"]   ← Streamlit Cloud Secrets
      2. os.environ["GEMINI_API_KEY"]   ← local .env / shell export
    """
    try:
        from google import genai  # imported here so the app can load without it
    except ImportError:
        st.error(
            "❌ The `google-genai` package is not installed. "
            "Run `pip install google-genai` and restart the app."
        )
        st.stop()

    # Try Streamlit secrets first, fall back to environment variable
    api_key = st.secrets.get("GEMINI_API_KEY", None)
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", None)

    if not api_key:
        st.error(
            "🔑 **API key not found.**\n\n"
            "To use this application you need a **Google Gemini API key**.\n\n"
            "**Locally:** Add it to `.streamlit/secrets.toml`:\n"
            "```toml\nGEMINI_API_KEY = \"your-key-here\"\n```\n\n"
            "**Streamlit Cloud:** Go to your app → ⋮ → Settings → Secrets "
            "and add:\n"
            "```toml\nGEMINI_API_KEY = \"your-key-here\"\n```\n\n"
            "Get a free key at [Google AI Studio](https://aistudio.google.com/)."
        )
        st.stop()

    return genai.Client(api_key=api_key)


# ─────────────────────────────────────────────────────────────
# HELPER: extract text from an uploaded PDF
# ─────────────────────────────────────────────────────────────
def extract_text_from_resume(uploaded_file) -> str:
    """
    Extract text from an uploaded PDF or DOCX resume.
    """
    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp.flush()

            reader = PdfReader(tmp.name)
            text = ""

            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text

        return text.strip()

    elif file_name.endswith(".docx"):
        from docx import Document

        document = Document(uploaded_file)
        text = []

        for paragraph in document.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)

        return "\n".join(text).strip()

    else:
        return ""

# ─────────────────────────────────────────────────────────────
# HELPER: call Gemini to analyze the resume
# ─────────────────────────────────────────────────────────────

ANALYSIS_PROMPT_TEMPLATE = """
You are an expert technical recruiter.

Analyze the job description and the resume.

The job description contains up to three kinds of content:
1. REQUIRED qualifications — must-have skills/experience, often under
   headings like "Requirements", "Required Qualifications", "Must Have".
2. PREFERRED qualifications — nice-to-have skills/experience, often under
   headings like "Preferred Qualifications", "Nice to Have", "Bonus Points".
3. RESPONSIBILITIES — the duties/tasks the role involves, often under
   headings like "Responsibilities", "What You'll Do", "Key Duties".

First, extract the concrete items belonging to each of these three
categories from the JOB DESCRIPTION only.

Then check whether each item is supported by evidence in the RESUME.

Rules:
- Only evaluate items that actually appear in the job description.
- Do not include skills that appear only in the resume.
- Use semantic understanding when comparing the resume and job description.
- Do not assume that a related skill is automatically a match.
- If there is not enough evidence in the resume, mark it as Missing.
- Classify every item into exactly one category: "required", "preferred",
  or "responsibilities". If the job description does not clearly separate
  required vs. preferred, use your best judgment (default ambiguous
  must-have-sounding items to "required").
- Return ONLY valid JSON.
- Do not include markdown or explanations.

Use exactly this structure:

{{
    "required": [
        {{"requirement": "Python", "status": "Match"}},
        {{"requirement": "5+ years experience", "status": "Missing"}}
    ],
    "preferred": [
        {{"requirement": "AWS certification", "status": "Missing"}},
        {{"requirement": "Docker", "status": "Match"}}
    ],
    "responsibilities": [
        {{"requirement": "Lead a team of engineers", "status": "Match"}},
        {{"requirement": "Own the CI/CD pipeline", "status": "Missing"}}
    ],
    "recommendations": [
        "Add quantified achievements to your experience section.",
        "Include a section on your Machine Learning projects."
    ]
}}

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}
"""


# ── Weights used to build the transparent overall score ──
# Required qualifications count most, responsibilities count least, since
# meeting the responsibilities is a weaker/looser signal than meeting an
# explicit hard requirement. If the job description doesn't contain any
# items for a given category (e.g. no "Preferred" section at all), that
# category is dropped and the remaining weights are re-normalized so they
# still add up to 100% — the score is always a full 0–100 scale.
CATEGORY_WEIGHTS = {
    "required": 0.60,
    "preferred": 0.25,
    "responsibilities": 0.15,
}
CATEGORY_LABELS = {
    "required": "Required Qualifications",
    "preferred": "Preferred Qualifications",
    "responsibilities": "Responsibilities",
}

# ── Retry settings for transient Gemini server errors (e.g. 503 overload) ──
MAX_GEMINI_RETRIES = 3
RETRY_BASE_DELAY_SECONDS = 2


def _generate_with_retry(client, prompt: str):
    """
    Call Gemini, automatically retrying with exponential backoff when the
    failure looks transient (e.g. "503 UNAVAILABLE ... high demand").

    Non-transient errors (bad/invalid API key, malformed request, etc.) are
    raised immediately without retrying.
    """
    last_exc = None
    for attempt in range(MAX_GEMINI_RETRIES):
        try:
            return client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
        except Exception as exc:
            err_msg = str(exc).upper()
            is_transient = (
                "UNAVAILABLE" in err_msg
                or "503" in err_msg
                or "OVERLOADED" in err_msg
            )
            last_exc = exc
            if not is_transient or attempt == MAX_GEMINI_RETRIES - 1:
                raise
            time.sleep(RETRY_BASE_DELAY_SECONDS * (2 ** attempt))
    if last_exc is not None:
        raise last_exc

raise RuntimeError("Gemini request failed after all retry attempts.")


def analyze_resume(client, resume_text: str, job_description: str) -> dict:
    """
    Send the resume + job description to Gemini and return a parsed dict.

    Returns a dict with keys:
        required          dict  (label, matched, missing, total, percentage)
        preferred         dict  (label, matched, missing, total, percentage)
        responsibilities  dict  (label, matched, missing, total, percentage)
        score             float  (0–100, transparent weighted overall score)
        breakdown         list[dict]  (per-category weight/contribution, for
                                        display so the score is auditable)
        recommendations   list[str]
    """
    prompt = ANALYSIS_PROMPT_TEMPLATE.format(
        job_description=job_description,
        resume_text=resume_text,
    )

    response = _generate_with_retry(client, prompt)

    raw_text = response.text.strip()

    # Strip markdown code fences if the model wraps in ```json … ```
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        raw_text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    analysis = json.loads(raw_text)

    # ── Parse each category's items into matched / missing ──
    categories = {}
    for key in ("required", "preferred", "responsibilities"):
        items = analysis.get(key, [])
        matched = [i["requirement"] for i in items if i.get("status") == "Match"]
        missing = [i["requirement"] for i in items if i.get("status") == "Missing"]
        total = len(matched) + len(missing)
        percentage = (len(matched) / total * 100) if total > 0 else None
        categories[key] = {
            "label": CATEGORY_LABELS[key],
            "matched": matched,
            "missing": missing,
            "total": total,
            "percentage": percentage,
        }

    # ── Transparent weighted overall score ──
    active_weights = {
        key: CATEGORY_WEIGHTS[key]
        for key, data in categories.items()
        if data["total"] > 0
    }
    weight_sum = sum(active_weights.values())

    breakdown = []
    score = 0.0
    for key, data in categories.items():
        if data["total"] > 0:
            normalized_weight = active_weights[key] / weight_sum
            contribution = data["percentage"] * normalized_weight
            score += contribution
        else:
            normalized_weight = 0.0
            contribution = 0.0
        breakdown.append({
            "key": key,
            "label": data["label"],
            "matched_count": len(data["matched"]),
            "missing_count": len(data["missing"]),
            "percentage": data["percentage"],
            "base_weight": CATEGORY_WEIGHTS[key],
            "normalized_weight": normalized_weight,
            "contribution": contribution,
        })

    recommendations = analysis.get("recommendations", [])

    return {
        "required": categories["required"],
        "preferred": categories["preferred"],
        "responsibilities": categories["responsibilities"],
        "score": score,
        "breakdown": breakdown,
        "recommendations": recommendations,
    }


# ─────────────────────────────────────────────────────────────
# HELPER: render score badge
# ─────────────────────────────────────────────────────────────
def render_score(score: float):
    if score >= 70:
        cls = "score-high"
        label = "Strong Match"
    elif score >= 40:
        cls = "score-medium"
        label = "Partial Match"
    else:
        cls = "score-low"
        label = "Weak Match"

    st.markdown(f"""
    <div class="score-container">
        <div class="score-circle {cls}">{round(score)}%</div>
        <br/>
        <p style="font-size:1.1rem; font-weight:600; color:#2c3e50; margin-top:8px;">{label}</p>
        <p style="color:#6c7a93; font-size:0.9rem;">Based on job description requirements</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():

    # ── Hero banner ───────────────────────────────────────────
    st.markdown("""
    <div class="hero-banner">
        <h1>Is your resume relevant?</h1>
        <p>
            Upload your resume and paste the job description.<br/>
            And AI will instantly tell you which requirements you meet,
            which are missing and how to improve your application.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column input layout ───────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown(
            '<p class="section-label"><span class="step-badge">1</span>Upload Resume (PDF)</p>',
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
    label="",
    type=["pdf", "docx"],
    help=f"PDF or DOCX · max {MAX_PDF_SIZE_MB} MB",
)
        if uploaded_file:
            size_mb = uploaded_file.size / (1024 * 1024)
            if size_mb > MAX_PDF_SIZE_MB:
                st.error(
                    f"⚠️ File is {size_mb:.1f} MB. "
                    f"Maximum allowed size is {MAX_PDF_SIZE_MB} MB."
                )
                uploaded_file = None
            else:
                st.success(f"✅ **{uploaded_file.name}** ({size_mb:.2f} MB) — ready")

    with col_right:
        st.markdown(
            '<p class="section-label"><span class="step-badge">2</span>Paste Job Description</p>',
            unsafe_allow_html=True,
        )
        job_description = st.text_area(
            label="",
            placeholder=(
                "Paste the full job description here.\n\n"
                "Include the required skills, responsibilities, and qualifications "
                "for the most accurate analysis."
            ),
            height=220,
        )

    # ── Analyze button ────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    analyze_clicked = st.button(
        "Analyze Resume",
        type="primary"
    )

    # ── Validation ────────────────────────────────────────────
    if analyze_clicked:
        errors = []
        if not uploaded_file:
            errors.append("Please upload a PDF resume.")
        if not job_description or len(job_description.strip()) < 30:
            errors.append("Please paste a job description (at least 30 characters).")

        if errors:
            for e in errors:
                st.warning(f"⚠️ {e}")
            return

        # ── Extract PDF text ──────────────────────────────────
        with st.spinner("📖 Extracting text from your resume…"):
            try:
                resume_text = extract_text_from_resume(uploaded_file)
            except Exception as exc:
                st.error(
                    f"❌ Could not read the PDF file.\n\n"
                    f"Make sure the file is not password-protected or corrupted.\n\n"
                    f"Technical detail: `{exc}`"
                )
                return

        if not resume_text:
            st.error(
                "❌ No text could be extracted from this PDF. "
                "If it is a scanned image, try a text-based PDF instead."
            )
            return

        # ── Call Gemini ───────────────────────────────────────
        client = get_gemini_client()

        with st.spinner(f"🤖 Analyzing with Gemini ({GEMINI_MODEL})… this takes a few seconds…"):
            try:
                result = analyze_resume(client, resume_text, job_description.strip())
            except json.JSONDecodeError:
                st.error(
                    "❌ The AI returned an unexpected response format. "
                    "Please try again. If this keeps happening, the model "
                    "may be temporarily unavailable."
                )
                return
            except Exception as exc:
                err_msg = str(exc)
                if "API_KEY" in err_msg.upper() or "INVALID" in err_msg.upper():
                    st.error(
                        "❌ **Invalid API key.** "
                        "Please check your Gemini API key in Streamlit Secrets or "
                        "`.streamlit/secrets.toml` and try again."
                    )
                elif "QUOTA" in err_msg.upper() or "RATE" in err_msg.upper():
                    st.warning(
                        "⏳ **Rate limit reached.** "
                        "You have exceeded the Gemini API free-tier quota. "
                        "Please wait a moment and try again."
                    )
                elif "TIMEOUT" in err_msg.upper():
                    st.error(
                        "⏱️ **Request timed out.** "
                        "The AI took too long to respond. Please try again."
                    )
                elif (
                    "UNAVAILABLE" in err_msg.upper()
                    or "503" in err_msg
                    or "OVERLOADED" in err_msg.upper()
                ):
                    st.warning(
                        "🚦 **Gemini is temporarily overloaded.** "
                        "Google's servers are experiencing high demand for this "
                        "model right now. This app already retried automatically, "
                        "but it's still busy — please wait a few seconds and "
                        "click **Analyze Resume** again."
                    )
                else:
                    st.error(
                        f"❌ **AI analysis failed.**\n\n"
                        f"Technical detail: `{err_msg}`"
                    )
                return

        # ── Results ───────────────────────────────────────────
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("## 📊 Analysis Results")

        # Row 1: Overall score | per-category match counts
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Overall Score", f"{round(result['score'])}%")
        with sc2:
            req = result["required"]
            st.metric("🔒 Required",   f"{len(req['matched'])}/{req['total']}")
        with sc3:
            pref = result["preferred"]
            st.metric("⭐ Preferred",  f"{len(pref['matched'])}/{pref['total']}")
        with sc4:
            resp = result["responsibilities"]
            st.metric("🧩 Responsibilities", f"{len(resp['matched'])}/{resp['total']}")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Row 2: Score visual | transparent score breakdown
        r_left, r_right = st.columns([1, 2], gap="large")

        with r_left:
            render_score(result["score"])

        with r_right:
            st.markdown(
                "#### 🧮 How the Overall Score Was Calculated",
                help="Each category's match rate is weighted, then combined "
                     "into the overall score. Categories with no items in "
                     "the job description are excluded and the remaining "
                     "weights are rescaled to still total 100%.",
            )
            for cat in result["breakdown"]:
                if cat["percentage"] is None:
                    st.caption(f"**{cat['label']}** — not present in this job description (excluded from score).")
                    continue
                st.markdown(
                    f"**{cat['label']}** — {round(cat['percentage'])}% match "
                    f"({cat['matched_count']}/{cat['matched_count'] + cat['missing_count']}) "
                    f"× weight {round(cat['normalized_weight'] * 100)}% "
                    f"= **{round(cat['contribution'], 1)} pts**"
                )
                st.progress(cat["percentage"] / 100)

        # Row 3: Required vs. Preferred vs. Responsibilities — kept separate
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown("## 🔍 Detailed Breakdown")

        def render_category(title, help_text, category_data, icon_match="✅", icon_missing="❌"):
            st.markdown(f"#### {title}", help=help_text)
            if category_data["total"] == 0:
                st.info("This job description did not specify any items in this category.")
                return

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**{icon_match} Met**")
                if category_data["matched"]:
                    pills_html = "".join(
                        f'<span class="pill-match">{skill}</span>'
                        for skill in category_data["matched"]
                    )
                    st.markdown(f'<div class="pill-container">{pills_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("None met.")
            with col_b:
                st.markdown(f"**{icon_missing} Not Found**")
                if category_data["missing"]:
                    pills_html = "".join(
                        f'<span class="pill-missing">{skill}</span>'
                        for skill in category_data["missing"]
                    )
                    st.markdown(f'<div class="pill-container">{pills_html}</div>', unsafe_allow_html=True)
                else:
                    st.caption("None missing.")

        render_category(
            "🔒 Required Qualifications",
            "Must-have qualifications explicitly requested in the job description.",
            result["required"],
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        render_category(
            "⭐ Preferred Qualifications",
            "Nice-to-have qualifications from the job description's preferred/bonus section.",
            result["preferred"],
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        render_category(
            "🧩 Responsibilities",
            "Duties/tasks the role involves, checked against evidence of similar experience in the resume.",
            result["responsibilities"],
        )

        # Row 3: Recommendations
        if result["recommendations"]:
            st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
            st.markdown("## 💡 Improvement Recommendations")
            st.markdown(
                "*Suggested ways to strengthen your application for this role:*"
            )
            for i, rec in enumerate(result["recommendations"], 1):
                st.markdown(
                    f'<div class="recommendation-box">💡 <b>{i}.</b> {rec}</div>',
                    unsafe_allow_html=True,
                )

        # Row 4: Footer note
        st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
        st.caption(
            "🤖 Analysis powered by Google Gemini · Results are AI-generated and may not be perfect. "
            "Use as a guide, not a definitive assessment."
        )


if __name__ == "__main__":
    main()