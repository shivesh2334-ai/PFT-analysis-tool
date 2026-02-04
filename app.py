import streamlit as st
import google.generativeai as genai
from PIL import Image
import pdf2image
import pandas as pd
import json
import io

# Page configuration
st.set_page_config(
    page_title="AI PFT Analyzer",
    page_icon="🫁",
    layout="wide"
)

# --- UTILS ---
def clean_json_string(json_str):
    """Clean markdown formatting from JSON string"""
    if "```json" in json_str:
        json_str = json_str.replace("```json", "").replace("```", "")
    elif "```" in json_str:
        json_str = json_str.replace("```", "")
    return json_str.strip()

def process_uploaded_file(uploaded_file):
    """Convert upload to format compatible with Gemini"""
    try:
        if uploaded_file.type == "application/pdf":
            # Convert first page of PDF to image
            images = pdf2image.convert_from_bytes(uploaded_file.read())
            if images:
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='JPEG')
                return {'mime_type': 'image/jpeg', 'data': img_byte_arr.getvalue()}, images[0]
        else:
            # It's an image
            return {'mime_type': uploaded_file.type, 'data': uploaded_file.getvalue()}, Image.open(uploaded_file)
    except Exception as e:
        st.error(f"File processing error: {e}")
        return None, None
    return None, None

# --- AI FUNCTIONS ---

def extract_pft_values_gemini(image_parts, api_key):
    """Uses Gemini 1.5 Flash (Vision)"""
    genai.configure(api_key=api_key)
    
    # We try specific stable model names
    model_name = 'gemini-1.5-flash'
    
    try:
        model = genai.GenerativeModel(model_name)
        
        prompt = """
        Analyze this PFT report image. Extract these exact values. Return 0.0 if not found.
        Return ONLY raw JSON:
        {
            "FEV1": float (Liters), "FEV1_pred": float (%),
            "FVC": float (Liters), "FVC_pred": float (%),
            "FEV1_FVC": float (%),
            "TLC": float (Liters), "TLC_pred": float (%),
            "DLCO": float, "DLCO_pred": float (%)
        }
        """
        
        # Flash 1.5 is standard now, no need for complex fallbacks if lib is updated
        response = model.generate_content([prompt, image_parts[0]])
        return json.loads(clean_json_string(response.text))
        
    except Exception as e:
        st.error(f"Extraction Error using {model_name}: {str(e)}")
        return None

def analyze_results_gemini(data, api_key):
    """Uses Gemini 1.5 Pro (Reasoning)"""
    genai.configure(api_key=api_key)
    
    try:
        model = genai.GenerativeModel('gemini-1.5-pro')
        
        prompt = f"""
        Act as a Pulmonologist. Interpret these PFT results:
        {json.dumps(data)}
        
        Provide a structured Markdown report with:
        1. Pattern (Obstructive/Restrictive/Normal)
        2. Detailed Interpretation (Airflow, Volumes, Diffusion)
        3. Clinical Impression
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Analysis Error: {str(e)}"

# --- MAIN UI ---

# Session State
if 'pft_data' not in st.session_state:
    st.session_state.pft_data = {k: 0.0 for k in ['FEV1','FEV1_pred','FVC','FVC_pred','FEV1_FVC','TLC','TLC_pred','DLCO','DLCO_pred']}

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Gemini API Key", type="password")
    
    if st.button("🛠️ Debug: Check Available Models"):
        if not api_key:
            st.error("Enter API Key first")
        else:
            try:
                genai.configure(api_key=api_key)
                models = [m.name for m in genai.list_models()]
                st.write("Available Models:", models)
                if 'models/gemini-1.5-flash' in models:
                    st.success("✅ Gemini 1.5 Flash is available!")
                else:
                    st.error("❌ Gemini 1.5 Flash NOT found. Check requirements.txt")
            except Exception as e:
                st.error(f"API Error: {e}")

# Main Layout
st.title("🫁 AI PFT Analyzer")

uploaded_file = st.file_uploader("Upload Report", type=['jpg', 'png', 'pdf'])

if uploaded_file and api_key:
    # Processing
    img_data, display_img = process_uploaded_file(uploaded_file)
    
    if display_img:
        st.image(display_img, caption="Report Preview", width=400)
        
        if st.button("🚀 Extract Data"):
            with st.spinner("Extracting..."):
                result = extract_pft_values_gemini([img_data], api_key)
                if result:
                    st.session_state.pft_data.update(result)
                    st.success("Extracted!")

    # Editable Data Form
    st.subheader("Verify Data")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.caption("Spirometry")
        st.session_state.pft_data['FEV1'] = st.number_input("FEV1 (L)", value=float(st.session_state.pft_data.get('FEV1', 0.0)))
        st.session_state.pft_data['FVC'] = st.number_input("FVC (L)", value=float(st.session_state.pft_data.get('FVC', 0.0)))
        st.session_state.pft_data['FEV1_FVC'] = st.number_input("FEV1/FVC (%)", value=float(st.session_state.pft_data.get('FEV1_FVC', 0.0)))
    with c2:
        st.caption("Predicted %")
        st.session_state.pft_data['FEV1_pred'] = st.number_input("FEV1 %Pred", value=float(st.session_state.pft_data.get('FEV1_pred', 0.0)))
        st.session_state.pft_data['FVC_pred'] = st.number_input("FVC %Pred", value=float(st.session_state.pft_data.get('FVC_pred', 0.0)))
    with c3:
        st.caption("Other")
        st.session_state.pft_data['TLC_pred'] = st.number_input("TLC %Pred", value=float(st.session_state.pft_data.get('TLC_pred', 0.0)))
        st.session_state.pft_data['DLCO_pred'] = st.number_input("DLCO %Pred", value=float(st.session_state.pft_data.get('DLCO_pred', 0.0)))

    # Analysis
    if st.button("🧠 Analyze Clinical Findings"):
        with st.spinner("Analyzing..."):
            report = analyze_results_gemini(st.session_state.pft_data, api_key)
            st.markdown("---")
            st.markdown(report)

elif not api_key:
    st.warning("Please enter your Gemini API Key in the sidebar.")
