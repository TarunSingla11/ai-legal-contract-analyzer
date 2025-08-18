from flask import Flask, request, render_template, session, redirect, url_for, make_response
import os
import json
import re
import logging
import pdfplumber
from werkzeug.utils import secure_filename
import markdown2
import google.generativeai as genai
from dotenv import load_dotenv
from weasyprint import HTML
import joblib

# Load environment variables from .env file
load_dotenv()

# --- BASIC SETUP ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'a-fallback-super-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# --- MODEL AND API CONFIGURATION ---
try:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
    print("Google AI SDK configured successfully.")
except Exception as e:
    print(f"Error configuring Google AI SDK: {e}")

try:
    classifier_pipeline = joblib.load('models/clause_classifier_pipeline.pkl')
    logging.info("Clause classifier pipeline loaded successfully.")
except FileNotFoundError:
    logging.error("FATAL: 'clause_classifier_pipeline.pkl' not found. The analysis feature will fail.")
    classifier_pipeline = None
except Exception as e:
    logging.error(f"Error loading 'clause_classifier_pipeline.pkl': {e}")
    classifier_pipeline = None


# --- Helper and Analysis Functions ---

def extract_chunks(pdf_path: str, chunk_size: int = 1500, overlap: int = 200) -> tuple[list[str], str]:
    """Extracts text and splits it into robust, overlapping chunks."""
    try:
        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=1, y_tolerance=3)
                if page_text:
                    full_text += page_text + "\n"
        if not full_text.strip():
            return [], ""
        chunks = []
        start_index = 0
        while start_index < len(full_text):
            end_index = start_index + chunk_size
            chunks.append(full_text[start_index:end_index])
            start_index += chunk_size - overlap
        meaningful_chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
        logging.info(f"Extracted {len(meaningful_chunks)} chunks from the document.")
        return meaningful_chunks, full_text
    except Exception as e:
        logging.error(f"Error extracting content from {pdf_path}: {str(e)}")
        return [], ""

def call_gemini(prompt: str, model_name: str = "gemini-1.5-flash-latest"):
    """Calls the Google Gemini API and returns the text response."""
    try:
        model = genai.GenerativeModel(model_name)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        response = model.generate_content(prompt, safety_settings=safety_settings)
        return response.text
    except Exception as e:
        logging.error(f"Google Gemini API call failed: {e}")
        return json.dumps({"error": "The AI model API call failed.", "details": str(e)})


# ### NEW HYBRID FUNCTION ###
def hybrid_categorize_and_summarize_clauses(chunks: list[str]):
    """
    Combines local classification with a single bulk API call for summary and risk.
    """
    if not classifier_pipeline:
        return [{"error": "The clause classifier model is not loaded. Cannot perform analysis."}]

    # Step 1: Classify all chunks locally first (fast, no API calls)
    pre_processed_data = []
    for chunk in chunks:
        predicted_category = classifier_pipeline.predict([chunk])[0]
        pre_processed_data.append({
            "chunk_text": chunk,
            "type": predicted_category
        })

    # Step 2: Create a single, large prompt for the LLM
    # We send the pre-classified data to the LLM for the nuanced tasks.
    pre_processed_json = json.dumps(pre_processed_data, indent=2)
    prompt = f"""I have pre-processed and classified several contract chunks using a local model.
Your task is to perform a risk assessment and write a one-sentence summary for EACH chunk provided in the JSON array below.

**RULES:**
1.  For each JSON object in the input, add two new keys: "risk" and "summary".
2.  "risk": Assess the potential risk as "Low", "Medium", or "High".
3.  "summary": Write a concise, one-sentence summary of the chunk's main point.
4.  Your entire response MUST BE ONLY a valid JSON array `[...]` that contains the completed objects. Do not include any other text or explanations.

**Pre-classified Chunks:**
{pre_processed_json}
"""

    # Step 3: Make a single API call
    response_text = call_gemini(prompt)

    # Step 4: Parse the response and merge with original data
    try:
        # Check for our custom error format first
        try:
            potential_error = json.loads(response_text)
            if isinstance(potential_error, dict) and 'error' in potential_error:
                return [{"error": potential_error.get('details', 'Unknown API error')}]
        except json.JSONDecodeError:
            pass # This is expected for a successful, non-JSON object string

        # Extract the JSON array from the LLM's response
        clean_json_str = re.search(r'\[.*\]', response_text, re.DOTALL).group(0)
        llm_results = json.loads(clean_json_str)

        # Ensure we got a result for every chunk we sent
        if len(llm_results) != len(pre_processed_data):
            raise ValueError("The AI model returned a different number of items than were sent.")

        # Combine the local results with the LLM results
        final_results = []
        for i, local_data in enumerate(pre_processed_data):
            llm_data = llm_results[i]
            final_results.append({
                "chunk_preview": local_data["chunk_text"][:50].strip().replace('"', '\\"') + "...",
                "type": local_data["type"],
                "risk": llm_data.get("risk", "Unknown"),
                "summary": llm_data.get("summary", "Summary not generated.")
            })
        return final_results

    except (json.JSONDecodeError, AttributeError, ValueError) as e:
        logging.error(f"Failed to decode or process bulk response from Gemini: {e}\nResponse was: {response_text}")
        return [{"error": "The AI model returned an invalid analysis format."}]


def generate_executive_brief(categorized_json: str):
    """Generates an executive brief from the categorized JSON."""
    master_prompt = f"""As an expert legal counsel, write a professional executive brief for a CEO using the following structured contract analysis.
**Structured Analysis (JSON):**
{categorized_json}
**Your Task:**
Generate a 300-400 word executive brief in Markdown. The brief MUST include these sections:
1. **Overall Recommendation:** A clear, one-sentence conclusion.
2. **Key Business Terms:** A bulleted list of the most critical commercial terms.
3. **Top 3 Red Flags:** A numbered list of the highest-risk items.
4. **Missing Clauses of Note:** A bulleted list of any standard clauses that appear to be missing."""
    return call_gemini(master_prompt)


def generate_general_summary(full_text: str):
    """Generates a general summary of the full text."""
    prompt = f"""You are a professional summarization engine. Your task is to provide a high-quality, comprehensive summary of the following document.
Focus on identifying the main purpose, key topics, and any important conclusions or action items.
Present the output as a well-structured Markdown document.
**Document Text:**
---
{full_text[:30000]}
---"""
    return call_gemini(prompt)


# --- Main Application Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if 'report_data' in session: session.pop('report_data', None)
    analysis, summary_result = None, None
    
    if request.method == 'POST':
        file = request.files.get('contract_file')
        action = request.form.get('action')

        if not file or not file.filename:
            return render_template('index.html', analysis=None, summary_result=None, error="Please select a file to upload.")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Processing '{filename}' with action: '{action}'")
        
        chunks, full_text = extract_chunks(filepath) 
        
        if action == 'analyze':
            if chunks:
                # Call the new hybrid function
                categorized_data = hybrid_categorize_and_summarize_clauses(chunks[:20])
                
                if not (isinstance(categorized_data, list) and categorized_data and 'error' in categorized_data[0]):
                    categorized_json_str = json.dumps(categorized_data, indent=2)
                    executive_brief_md = generate_executive_brief(categorized_json_str)

                    if 'error' in executive_brief_md.lower():
                        analysis = {'error': executive_brief_md, 'filename': filename}
                    else:
                        analysis = {
                            'filename': filename,
                            'executive_brief_html': markdown2.markdown(executive_brief_md, extras=["tables", "fenced-code-blocks"]),
                            'detailed_analysis': categorized_data
                        }
                        session['report_data'] = {'analysis': analysis, 'filename': filename}
                else:
                    analysis = {'error': categorized_data[0]['error'], 'filename': filename}
            else:
                analysis = {'error': 'Could not extract any meaningful text chunks for analysis.', 'filename': filename}

        elif action == 'summarize':
            if full_text:
                summary_md = generate_general_summary(full_text)
                if 'error' not in summary_md.lower():
                    summary_result = {
                        'filename': filename,
                        'summary_html': markdown2.markdown(summary_md, extras=["tables", "fenced-code-blocks"])
                    }
                    session['report_data'] = {'summary_result': summary_result, 'filename': filename}
                else:
                    summary_result = {'error': summary_md, 'filename': filename}
            else:
                summary_result = {'error': 'Could not extract any text from the document to summarize.', 'filename': filename}

    return render_template('index.html', analysis=analysis, summary_result=summary_result)


@app.route('/download_pdf')
def download_pdf():
    """Generates and serves a PDF report."""
    report_data = session.get('report_data')
    if not report_data: return redirect(url_for('index'))
    html_for_pdf = render_template('report_template.html', **report_data)
    pdf = HTML(string=html_for_pdf).write_pdf()
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=Aura_Report.pdf'
    return response

if __name__ == '__main__':
    app.run(debug=True)