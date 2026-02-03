import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdf2image
import pandas as pd
import json
import io

# Page configuration
st.set_page_config(
    page_title="AI PFT Analyzer (Powered by Gemini)",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #0d47a1;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #1e88e5;
        padding-bottom: 0.5rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pft_data' not in st.session_state:
    st.session_state.pft_data = {
        'FEV1': 0.0, 'FEV1_pred': 0.0,
        'FVC': 0.0, 'FVC_pred': 0.0,
        'FEV1_FVC': 0.0,
        'TLC': 0.0, 'TLC_pred': 0.0,
        'DLCO': 0.0, 'DLCO_pred': 0.0
    }
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = ""

def clean_json_string(json_str):
    """Clean markdown formatting from JSON string"""
    if "```json" in json_str:
        json_str = json_str.replace("```json", "").replace("```", "")
    elif "```" in json_str:
        json_str = json_str.replace("```", "")
    return json_str.strip()

def extract_pft_values_gemini(image_parts, api_key):
    """
    Uses Gemini 1.5 Flash to extract data from the image.
    Flash is faster and cheaper for extraction tasks.
    """
    try:
        genai.configure(api_key=api_key)
        # Using 1.5 Flash for speed and vision capabilities
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        prompt = """
        You are a medical data extraction assistant. 
        Analyze this Pulmonary Function Test (PFT) report image.
        Extract the following values specifically. If a value is not present, return 0.0.
        
        Return ONLY a raw JSON object (no markdown formatting) with these exact keys:
        {
            "FEV1": float (Liters, actual/observed),
            "FEV1_pred": float (% predicted),
            "FVC": float (Liters, actual/observed),
            "FVC_pred": float (% predicted),
            "FEV1_FVC": float (ratio percentage, e.g., 75.5 for 75.5%),
            "TLC": float (Liters, actual/observed),
            "TLC_pred": float (% predicted),
            "DLCO": float (actual/observed),
            "DLCO_pred": float (% predicted)
        }
        
        Look closely at the columns. Usually labeled "Pre", "Observed", "Actual" vs "Ref", "Pred", "%Pred".
        """
        
        response = model.generate_content([prompt, image_parts[0]])
        cleaned_json = clean_json_string(response.text)
        return json.loads(cleaned_json)
        
    except Exception as e:
        st.error(f"Extraction Error: {str(e)}")
        return None

def analyze_results_gemini(data, api_key):
    """
    Uses Gemini 1.5 Pro for medical reasoning and interpretation.
    """
    try:
        genai.configure(api_key=api_key)
        # Using 1.5 Pro for better reasoning capabilities
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        You are an expert Pulmonologist (MedGemma equivalent). 
        Analyze the following Pulmonary Function Test (PFT) data and provide a structured clinical interpretation.
        
        DATA:
        {json.dumps(data, indent=2)}
        
        GUIDELINES:
        1. **Spirometry**: Evaluate for Obstruction (FEV1/FVC < 70% or LLN), Restriction (FVC < 80%), or Mixed patterns. Grade severity based on FEV1 % pred.
        2. **Lung Volumes**: Confirm restriction if TLC < 80%. Check for hyperinflation/air trapping if TLC/RV are high.
        3. **Diffusion**: Evaluate DLCO. (Normal > 80% pred).
        
        OUTPUT FORMAT (Markdown):
        ### 1. Technical Quality & Pattern
        [Concise summary of the pattern]
        
        ### 2. Detailed Interpretation
        *   **Airflow:** [Analysis]
        *   **Volumes:** [Analysis]
        *   **Gas Exchange:** [Analysis]
        
        ### 3. Impression & Severity
        [Final diagnosis suggestion, e.g., "Moderate Obstructive Ventilatory Defect"]
        
        ### 4. Differential Diagnosis & Recommendations
        [List potential causes and next steps]
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis Error: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Convert upload to format compatible with Gemini"""
    if uploaded_file.type == "application/pdf":
        try:
            # Convert first page of PDF to image
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            if images:
                # Convert PIL image to bytes
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='JPEG')
                return {'mime_type': 'image/jpeg', 'data': img_byte_arr.getvalue()}, images[0]
        except Exception as e:
            st.error("Error processing PDF. Ensure poppler is installed or try uploading an Image.")
            return None, None
    else:
        # It's an image
        return {'mime_type': uploaded_file.type, 'data': uploaded_file.getvalue()}, Image.open(uploaded_file)
    return None, None

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    api_key = st.text_input("Google Gemini API Key", type="password", help="Get your key from Google AI Studio")
    
    if not api_key:
        st.warning("⚠️ API Key required for AI features")
        st.markdown("[Get API Key](https://aistudio.google.com/app/apikey)")
    
    st.markdown("---")
    st.info("""
    **How it works:**
    1. **Gemini 1.5 Flash**: Reads the numbers from your image/PDF (OCR).
    2. **Gemini 1.5 Pro**: Acts as a specialist to interpret the clinical findings.
    """)
    
    st.markdown("---")
    with st.expander("Values Reference"):
        st.markdown("""
        * **FEV1**: Forced Expiratory Volume in 1s
        * **FVC**: Forced Vital Capacity
        * **TLC**: Total Lung Capacity
        * **DLCO**: Diffusion Capacity
        """)

# Main Content
st.markdown('<h1 class="main-header">🫁 AI Pulmonary Function Analysis</h1>', unsafe_allow_html=True)
st.markdown('<div style="text-align: center;">Powered by Google Gemini 1.5 Pro & Flash</div>', unsafe_allow_html=True)

# Step 1: Upload
st.markdown('<div class="section-header">1. Upload Report</div>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload PFT Report (Image or PDF)", type=['jpg', 'jpeg', 'png', 'pdf'])

image_data = None
display_image = None

if uploaded_file:
    image_data, display_image = process_uploaded_file(uploaded_file)
    with col1:
        if display_image:
            st.image(display_image, caption="Uploaded Report", use_container_width=True)

with col2:
    st.info("Upload a report to enable extraction.")
    if uploaded_file and api_key:
        if st.button("🚀 AI Extract Values", type="primary"):
            with st.spinner("Gemini is reading the document..."):
                extracted_data = extract_pft_values_gemini([image_data], api_key)
                if extracted_data:
                    st.session_state.pft_data.update(extracted_data)
                    st.success("Extraction Complete! Please verify values below.")
    elif uploaded_file and not api_key:
        st.error("Please enter an API Key in the sidebar.")

# Step 2: Verify Data
st.markdown('<div class="section-header">2. Verify & Edit Data</div>', unsafe_allow_html=True)
st.caption("AI extraction is powerful but not perfect. Please verify these numbers against the report before analyzing.")

input_col1, input_col2, input_col3 = st.columns(3)

with input_col1:
    st.subheader("Spirometry")
    st.session_state.pft_data['FEV1'] = st.number_input("FEV1 (L)", value=float(st.session_state.pft_data.get('FEV1', 0.0)))
    st.session_state.pft_data['FEV1_pred'] = st.number_input("FEV1 % Pred", value=float(st.session_state.pft_data.get('FEV1_pred', 0.0)))
    st.session_state.pft_data['FVC'] = st.number_input("FVC (L)", value=float(st.session_state.pft_data.get('FVC', 0.0)))
    st.session_state.pft_data['FVC_pred'] = st.number_input("FVC % Pred", value=float(st.session_state.pft_data.get('FVC_pred', 0.0)))
    st.session_state.pft_data['FEV1_FVC'] = st.number_input("FEV1/FVC Ratio (%)", value=float(st.session_state.pft_data.get('FEV1_FVC', 0.0)))

with input_col2:
    st.subheader("Lung Volumes")
    st.session_state.pft_data['TLC'] = st.number_input("TLC (L)", value=float(st.session_state.pft_data.get('TLC', 0.0)))
    st.session_state.pft_data['TLC_pred'] = st.number_input("TLC % Pred", value=float(st.session_state.pft_data.get('TLC_pred', 0.0)))

with input_col3:
    st.subheader("Diffusion")
    st.session_state.pft_data['DLCO'] = st.number_input("DLCO", value=float(st.session_state.pft_data.get('DLCO', 0.0)))
    st.session_state.pft_data['DLCO_pred'] = st.number_input("DLCO % Pred", value=float(st.session_state.pft_data.get('DLCO_pred', 0.0)))

# Step 3: Analysis
st.markdown('<div class="section-header">3. Clinical Interpretation</div>', unsafe_allow_html=True)

if st.button("🧠 Generate Expert Analysis"):
    if not api_key:
        st.error("Please provide an API Key in the sidebar.")
    else:
        # Check if data is empty
        if st.session_state.pft_data['FEV1_FVC'] == 0 and st.session_state.pft_data['FVC_pred'] == 0:
            st.warning("⚠️ The values seem to be empty. Please enter data or upload a report.")
        else:
            with st.spinner("Consulting MedGemma model..."):
                analysis = analyze_results_gemini(st.session_state.pft_data, api_key)
                st.session_state.analysis_result = analysis

if st.session_state.analysis_result:
    st.markdown("---")
    st.markdown(st.session_state.analysis_result)
    
    # Disclaimer
    st.warning("""
    **Disclaimer:** This tool uses Artificial Intelligence (Gemini 1.5) to assist in interpretation. 
    It is not a medical device and should not replace professional medical advice. 
    Always verify results with the original report and clinical presentation.
    """)
