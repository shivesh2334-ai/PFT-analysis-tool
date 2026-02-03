import streamlit as st
import pytesseract
from PIL import Image
import pdf2image
import pandas as pd
import re
import os
import google.generativeai as genai
from io import BytesIO

# Page configuration
st.set_page_config(
    page_title="PFT Analysis Tool",
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
    .interpretation-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .normal-result {
        color: #2e7d32;
        font-weight: bold;
    }
    .abnormal-result {
        color: #c62828;
        font-weight: bold;
    }
    .warning-result {
        color: #f57c00;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pft_data' not in st.session_state:
    st.session_state.pft_data = {}
if 'interpretation_done' not in st.session_state:
    st.session_state.interpretation_done = False

def extract_text_from_image(image):
    """Extract text from image using OCR"""
    try:
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        st.error(f"Error extracting text from image: {str(e)}")
        return ""

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file"""
    try:
        images = pdf2image.convert_from_bytes(pdf_file.read())
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return ""

def parse_pft_values(text):
    """Parse PFT values from extracted text"""
    pft_data = {}
    
    # Common PFT parameters patterns
    patterns = {
        'FVC': r'FVC[:\s]+(\d+\.?\d*)',
        'FEV1': r'FEV1[:\s]+(\d+\.?\d*)',
        'FEV1/FVC': r'FEV1/FVC[:\s]+(\d+\.?\d*)',
        'TLC': r'TLC[:\s]+(\d+\.?\d*)',
        'RV': r'RV[:\s]+(\d+\.?\d*)',
        'DLCO': r'DLCO[:\s]+(\d+\.?\d*)',
        'FEF25-75': r'FEF25-75[:\s]+(\d+\.?\d*)',
    }
    
    # Predicted value patterns
    pred_patterns = {
        'FVC_pred': r'FVC[^0-9]*(\d+\.?\d*)[^0-9]*%',
        'FEV1_pred': r'FEV1[^0-9]*(\d+\.?\d*)[^0-9]*%',
        'TLC_pred': r'TLC[^0-9]*(\d+\.?\d*)[^0-9]*%',
        'DLCO_pred': r'DLCO[^0-9]*(\d+\.?\d*)[^0-9]*%',
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pft_data[key] = float(match.group(1))
    
    for key, pattern in pred_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            pft_data[key] = float(match.group(1))
    
    return pft_data

def classify_severity(value, parameter):
    """Classify the severity based on percentage of predicted"""
    if parameter in ['FVC_pred', 'FEV1_pred', 'DLCO_pred']:
        if value >= 80:
            return "Normal", "normal"
        elif value >= 70:
            return "Mild", "warning"
        elif value >= 60:
            return "Moderate", "warning"
        elif value >= 50:
            return "Moderately Severe", "abnormal"
        elif value >= 35:
            return "Severe", "abnormal"
        else:
            return "Very Severe", "abnormal"
    return "Unknown", "normal"

def interpret_spirometry(fev1_fvc, fev1_pred, fvc_pred):
    """Interpret spirometry results"""
    interpretation = []
    pattern = "Normal"
    
    if fev1_fvc < 70:
        pattern = "Obstructive"
        interpretation.append("**Obstructive Pattern Detected:**")
        interpretation.append(f"- FEV1/FVC ratio is {fev1_fvc:.1f}% (< 70%)")
        
        if fev1_pred:
            severity, _ = classify_severity(fev1_pred, 'FEV1_pred')
            interpretation.append(f"- Severity: {severity} obstruction (FEV1 {fev1_pred:.1f}% predicted)")
    
    elif fvc_pred and fvc_pred < 80 and (not fev1_fvc or fev1_fvc >= 70):
        pattern = "Restrictive"
        interpretation.append("**Restrictive Pattern Suggested:**")
        interpretation.append(f"- FVC is {fvc_pred:.1f}% of predicted (< 80%)")
        interpretation.append(f"- FEV1/FVC ratio is preserved ({fev1_fvc:.1f}% if available)")
        interpretation.append("- Consider lung volume measurements for confirmation")
    
    elif fev1_fvc < 70 and fvc_pred and fvc_pred < 80:
        pattern = "Mixed"
        interpretation.append("**Mixed Pattern (Obstructive + Restrictive):**")
        interpretation.append(f"- FEV1/FVC ratio is {fev1_fvc:.1f}% (< 70%) - Obstruction")
        interpretation.append(f"- FVC is {fvc_pred:.1f}% of predicted (< 80%) - Restriction")
    
    else:
        interpretation.append("**Normal Spirometry Pattern:**")
        interpretation.append(f"- FEV1/FVC ratio: {fev1_fvc:.1f}% (≥ 70%)")
        if fvc_pred:
            interpretation.append(f"- FVC: {fvc_pred:.1f}% of predicted (≥ 80%)")
        if fev1_pred:
            interpretation.append(f"- FEV1: {fev1_pred:.1f}% of predicted (≥ 80%)")
    
    return pattern, interpretation

def interpret_lung_volumes(tlc_pred, rv_pred):
    """Interpret lung volume measurements"""
    interpretation = []
    
    if not tlc_pred:
        return ["Lung volume data not available"]
    
    if tlc_pred < 80:
        interpretation.append("**Restrictive Pattern Confirmed:**")
        interpretation.append(f"- TLC is {tlc_pred:.1f}% of predicted (< 80%)")
        severity, _ = classify_severity(tlc_pred, 'TLC_pred')
        interpretation.append(f"- Severity: {severity} restriction")
    elif tlc_pred > 120:
        interpretation.append("**Hyperinflation Detected:**")
        interpretation.append(f"- TLC is {tlc_pred:.1f}% of predicted (> 120%)")
    else:
        interpretation.append("**Normal Lung Volumes:**")
        interpretation.append(f"- TLC is {tlc_pred:.1f}% of predicted (normal range)")
    
    if rv_pred:
        if rv_pred > 120:
            interpretation.append(f"- Increased residual volume (RV {rv_pred:.1f}% predicted) - suggests air trapping")
        elif rv_pred < 80:
            interpretation.append(f"- Decreased residual volume (RV {rv_pred:.1f}% predicted)")
        else:
            interpretation.append(f"- Normal residual volume (RV {rv_pred:.1f}% predicted)")
    
    return interpretation

def interpret_dlco(dlco_pred):
    """Interpret DLCO results"""
    interpretation = []
    
    if not dlco_pred:
        return ["DLCO data not available"]
    
    if dlco_pred < 80:
        severity, _ = classify_severity(dlco_pred, 'DLCO_pred')
        interpretation.append("**Reduced Diffusion Capacity:**")
        interpretation.append(f"- DLCO is {dlco_pred:.1f}% of predicted (< 80%)")
        interpretation.append(f"- Severity: {severity}")
        interpretation.append("- May indicate:")
        interpretation.append("  - Emphysema/COPD")
        interpretation.append("  - Interstitial lung disease")
        interpretation.append("  - Pulmonary vascular disease")
        interpretation.append("  - Anemia (if present)")
    elif dlco_pred > 120:
        interpretation.append("**Increased Diffusion Capacity:**")
        interpretation.append(f"- DLCO is {dlco_pred:.1f}% of predicted (> 120%)")
        interpretation.append("- May indicate:")
        interpretation.append("  - Polycythemia")
        interpretation.append("  - Alveolar hemorrhage")
        interpretation.append("  - Left-to-right cardiac shunt")
    else:
        interpretation.append("**Normal Diffusion Capacity:**")
        interpretation.append(f"- DLCO is {dlco_pred:.1f}% of predicted (normal range)")
    
    return interpretation

def generate_final_impression(pattern, pft_data):
    """Generate final clinical impression"""
    impression = []
    
    # Main pattern
    impression.append(f"**Primary Pattern: {pattern}**")
    impression.append("")
    
    # Detailed findings
    if pattern == "Obstructive":
        impression.append("Findings consistent with obstructive lung disease.")
        if pft_data.get('DLCO_pred') and pft_data['DLCO_pred'] < 80:
            impression.append("Reduced DLCO suggests parenchymal involvement (e.g., emphysema).")
        impression.append("")
        impression.append("**Differential Diagnosis:**")
        impression.append("- Chronic Obstructive Pulmonary Disease (COPD)")
        impression.append("- Asthma")
        impression.append("- Bronchiectasis")
        impression.append("- Chronic bronchitis")
    
    elif pattern == "Restrictive":
        impression.append("Findings suggest restrictive lung disease.")
        impression.append("")
        impression.append("**Differential Diagnosis:**")
        impression.append("- Interstitial lung disease (ILD)")
        impression.append("- Chest wall disorders")
        impression.append("- Neuromuscular disease")
        impression.append("- Obesity")
        impression.append("- Pleural disease")
    
    elif pattern == "Mixed":
        impression.append("Findings show both obstructive and restrictive components.")
        impression.append("")
        impression.append("**Differential Diagnosis:**")
        impression.append("- Combined pulmonary fibrosis and emphysema (CPFE)")
        impression.append("- COPD with concurrent ILD")
        impression.append("- Severe asthma with air trapping")
    
    else:
        impression.append("Pulmonary function tests are within normal limits.")
    
    # Recommendations
    impression.append("")
    impression.append("**Recommendations:**")
    impression.append("- Clinical correlation is essential")
    impression.append("- Consider chest imaging if not already performed")
    if pattern != "Normal":
        impression.append("- Follow-up PFTs may be indicated to assess progression or response to treatment")
        impression.append("- Consider referral to pulmonologist if not already involved")
    
    return impression

def get_ai_review(pft_data, pattern, api_key):
    """Get AI review using Gemini API"""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        
        # Prepare the data summary
        data_summary = f"""
        Please review the following Pulmonary Function Test (PFT) results and provide a clinical interpretation:
        
        PFT Data:
        {pd.DataFrame([pft_data]).to_string()}
        
        Automated Pattern Analysis: {pattern}
        
        Please provide:
        1. Confirmation or refinement of the pattern identification
        2. Clinical significance of the findings
        3. Additional considerations or differential diagnoses
        4. Recommendations for further evaluation if needed
        """
        
        response = model.generate_content(data_summary)
        return response.text
    except Exception as e:
        return f"Error getting AI review: {str(e)}"

# Main App
st.markdown('<h1 class="main-header">🫁 Pulmonary Function Test (PFT) Analysis Tool</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This tool helps analyze Pulmonary Function Test (PFT) results.
    
    **Features:**
    - Upload PFT reports (Image/PDF)
    - OCR extraction
    - Manual data entry/review
    - Step-by-step interpretation
    - AI-powered review (Gemini)
    """)
    
    st.header("Instructions")
    st.markdown("""
    1. Upload your PFT report
    2. Review extracted values
    3. View detailed interpretation
    4. Optional: Get AI review
    """)

# Main content
tab1, tab2, tab3 = st.tabs(["📤 Upload & Extract", "📊 Analysis & Interpretation", "🤖 AI Review"])

with tab1:
    st.markdown('<div class="section-header">Step 1: Upload PFT Report</div>', unsafe_allow_html=True)
    
    input_method = st.radio("Choose input method:", ["Upload File", "Manual Entry"])
    
    if input_method == "Upload File":
        uploaded_file = st.file_uploader("Upload PFT report (JPG, JPEG, PNG, PDF)", 
                                        type=['jpg', 'jpeg', 'png', 'pdf'])
        
        if uploaded_file is not None:
            file_type = uploaded_file.type
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("Uploaded Document")
                if 'image' in file_type:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded PFT Report", use_container_width=True)
                elif 'pdf' in file_type:
                    st.success("PDF uploaded successfully")
                    st.info("Click 'Extract Values' to process the PDF")
            
            with col2:
                st.subheader("Extraction Options")
                if st.button("🔍 Extract Values from Document", type="primary"):
                    with st.spinner("Extracting text from document..."):
                        if 'image' in file_type:
                            image = Image.open(uploaded_file)
                            text = extract_text_from_image(image)
                        elif 'pdf' in file_type:
                            text = extract_text_from_pdf(uploaded_file)
                        
                        if text:
                            st.success("Text extracted successfully!")
                            with st.expander("View extracted text"):
                                st.text(text)
                            
                            # Parse values
                            parsed_data = parse_pft_values(text)
                            if parsed_data:
                                st.session_state.pft_data = parsed_data
                                st.success(f"Extracted {len(parsed_data)} PFT values!")
                            else:
                                st.warning("No PFT values automatically detected. Please enter manually below.")
    
    st.markdown('<div class="section-header">Step 2: Review/Enter PFT Values</div>', unsafe_allow_html=True)
    
    st.info("Review extracted values or enter manually. Enter values as percentage of predicted where applicable.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Spirometry")
        fev1 = st.number_input("FEV1 (L)", value=float(st.session_state.pft_data.get('FEV1', 0.0)), 
                               min_value=0.0, step=0.1, format="%.2f")
        fev1_pred = st.number_input("FEV1 (% predicted)", 
                                    value=float(st.session_state.pft_data.get('FEV1_pred', 0.0)),
                                    min_value=0.0, max_value=200.0, step=1.0, format="%.1f")
        
        fvc = st.number_input("FVC (L)", value=float(st.session_state.pft_data.get('FVC', 0.0)),
                              min_value=0.0, step=0.1, format="%.2f")
        fvc_pred = st.number_input("FVC (% predicted)", 
                                   value=float(st.session_state.pft_data.get('FVC_pred', 0.0)),
                                   min_value=0.0, max_value=200.0, step=1.0, format="%.1f")
        
        fev1_fvc = st.number_input("FEV1/FVC (%)", 
                                   value=float(st.session_state.pft_data.get('FEV1/FVC', 0.0)),
                                   min_value=0.0, max_value=100.0, step=0.1, format="%.1f")
    
    with col2:
        st.subheader("Lung Volumes")
        tlc = st.number_input("TLC (L)", value=float(st.session_state.pft_data.get('TLC', 0.0)),
                              min_value=0.0, step=0.1, format="%.2f")
        tlc_pred = st.number_input("TLC (% predicted)", 
                                   value=float(st.session_state.pft_data.get('TLC_pred', 0.0)),
                                   min_value=0.0, max_value=200.0, step=1.0, format="%.1f")
        
        rv = st.number_input("RV (L)", value=float(st.session_state.pft_data.get('RV', 0.0)),
                             min_value=0.0, step=0.1, format="%.2f")
        rv_pred = st.number_input("RV (% predicted)", 
                                  value=float(st.session_state.pft_data.get('RV_pred', 0.0)),
                                  min_value=0.0, max_value=200.0, step=1.0, format="%.1f")
    
    with col3:
        st.subheader("Diffusion Capacity")
        dlco = st.number_input("DLCO", value=float(st.session_state.pft_data.get('DLCO', 0.0)),
                               min_value=0.0, step=0.1, format="%.2f")
        dlco_pred = st.number_input("DLCO (% predicted)", 
                                    value=float(st.session_state.pft_data.get('DLCO_pred', 0.0)),
                                    min_value=0.0, max_value=200.0, step=1.0, format="%.1f")
    
    if st.button("💾 Save Values and Generate Interpretation", type="primary"):
        st.session_state.pft_data = {
            'FEV1': fev1,
            'FEV1_pred': fev1_pred,
            'FVC': fvc,
            'FVC_pred': fvc_pred,
            'FEV1/FVC': fev1_fvc,
            'TLC': tlc,
            'TLC_pred': tlc_pred,
            'RV': rv,
            'RV_pred': rv_pred,
            'DLCO': dlco,
            'DLCO_pred': dlco_pred
        }
        st.session_state.interpretation_done = True
        st.success("✅ Values saved! Go to 'Analysis & Interpretation' tab to view results.")

with tab2:
    st.markdown('<div class="section-header">Detailed PFT Interpretation</div>', unsafe_allow_html=True)
    
    if not st.session_state.interpretation_done or not st.session_state.pft_data:
        st.warning("⚠️ Please enter PFT values in the 'Upload & Extract' tab first.")
    else:
        data = st.session_state.pft_data
        
        # Display entered values
        st.subheader("📋 Entered Values Summary")
        df = pd.DataFrame([data])
        st.dataframe(df, use_container_width=True)
        
        # Step-by-step interpretation
        st.markdown('<div class="section-header">Step-by-Step Interpretation</div>', unsafe_allow_html=True)
        
        # Step 1: Spirometry
        st.markdown("### 1️⃣ Spirometry Analysis")
        pattern, spiro_interp = interpret_spirometry(
            data.get('FEV1/FVC', 0),
            data.get('FEV1_pred', 0),
            data.get('FVC_pred', 0)
        )
        
        for line in spiro_interp:
            st.markdown(line)
        
        # Severity classification
        if data.get('FEV1_pred'):
            severity, severity_class = classify_severity(data['FEV1_pred'], 'FEV1_pred')
            if severity_class == "normal":
                st.markdown(f'<p class="normal-result">Severity: {severity}</p>', unsafe_allow_html=True)
            elif severity_class == "warning":
                st.markdown(f'<p class="warning-result">Severity: {severity}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="abnormal-result">Severity: {severity}</p>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Step 2: Lung Volumes
        st.markdown("### 2️⃣ Lung Volume Analysis")
        vol_interp = interpret_lung_volumes(
            data.get('TLC_pred', None),
            data.get('RV_pred', None)
        )
        
        for line in vol_interp:
            st.markdown(line)
        
        st.markdown("---")
        
        # Step 3: DLCO
        st.markdown("### 3️⃣ Diffusion Capacity Analysis")
        dlco_interp = interpret_dlco(data.get('DLCO_pred', None))
        
        for line in dlco_interp:
            st.markdown(line)
        
        st.markdown("---")
        
        # Final Impression
        st.markdown('<div class="section-header">Final Clinical Impression</div>', unsafe_allow_html=True)
        
        impression = generate_final_impression(pattern, data)
        
        st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
        for line in impression:
            st.markdown(line)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Download report
        st.markdown("---")
        report_text = f"""
PULMONARY FUNCTION TEST INTERPRETATION REPORT
{'='*60}

PATIENT DATA:
{df.to_string()}

SPIROMETRY ANALYSIS:
{chr(10).join(spiro_interp)}

LUNG VOLUME ANALYSIS:
{chr(10).join(vol_interp)}

DIFFUSION CAPACITY ANALYSIS:
{chr(10).join(dlco_interp)}

FINAL IMPRESSION:
{chr(10).join(impression)}

{'='*60}
Report generated by PFT Analysis Tool
"""
        
        st.download_button(
            label="📥 Download Interpretation Report",
            data=report_text,
            file_name="pft_interpretation_report.txt",
            mime="text/plain"
        )

with tab3:
    st.markdown('<div class="section-header">AI-Powered Review (Gemini)</div>', unsafe_allow_html=True)
    
    if not st.session_state.interpretation_done:
        st.warning("⚠️ Please complete the interpretation in the 'Analysis & Interpretation' tab first.")
    else:
        st.info("💡 Get an AI-powered second opinion on your PFT interpretation using Google's Gemini API.")
        
        api_key = st.text_input("Enter your Gemini API Key:", type="password",
                                help="Get your free API key from https://makersuite.google.com/app/apikey")
        
        if st.button("🤖 Get AI Review", type="primary"):
            if not api_key:
                st.error("Please enter your Gemini API key.")
            else:
                with st.spinner("Analyzing with AI... This may take a moment."):
                    # Get pattern from previous analysis
                    data = st.session_state.pft_data
                    pattern, _ = interpret_spirometry(
                        data.get('FEV1/FVC', 0),
                        data.get('FEV1_pred', 0),
                        data.get('FVC_pred', 0)
                    )
                    
                    ai_review = get_ai_review(data, pattern, api_key)
                    
                    st.markdown("### 🤖 AI Analysis Results")
                    st.markdown('<div class="interpretation-box">', unsafe_allow_html=True)
                    st.markdown(ai_review)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.info("**Note:** AI review is for educational purposes only. Always consult with a qualified healthcare professional for clinical decisions.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational and informational purposes only. 
    It is not a substitute for professional medical advice, diagnosis, or treatment.</p>
    <p>Always consult with a qualified healthcare provider regarding any medical condition.</p>
</div>
""", unsafe_allow_html=True)
