# 🫁 Pulmonary Function Test (PFT) Analysis Tool

A comprehensive web application built with Streamlit for analyzing Pulmonary Function Test (PFT) results. This tool provides automated interpretation, step-by-step analysis, and AI-powered review using Google's Gemini API.

## 🌟 Features

- **📤 Multiple Input Methods**
  - Upload PFT reports as images (JPG, JPEG, PNG)
  - Upload PDF reports
  - Manual data entry option

- **🔍 Intelligent OCR Extraction**
  - Automatic text extraction from images using Tesseract OCR
  - PDF text extraction
  - Automatic parsing of PFT values

- **📊 Comprehensive Analysis**
  - Spirometry interpretation (FEV1, FVC, FEV1/FVC ratio)
  - Lung volume analysis (TLC, RV)
  - Diffusion capacity assessment (DLCO)
  - Severity classification
  - Pattern identification (Obstructive, Restrictive, Mixed, Normal)

- **📋 Step-by-Step Interpretation**
  - Detailed explanation of each parameter
  - Clinical significance
  - Differential diagnoses
  - Treatment recommendations

- **🤖 AI-Powered Review**
  - Integration with Google Gemini AI
  - Second opinion on interpretation
  - Additional clinical insights

- **📥 Report Generation**
  - Downloadable interpretation reports
  - Professional formatting
  - Complete analysis summary

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Tesseract OCR installed on your system
- Google Gemini API key (optional, for AI review feature)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/pft-analysis-tool.git
cd pft-analysis-tool
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Install Tesseract OCR**

**For Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**For macOS:**
```bash
brew install tesseract
```

**For Windows:**
- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to your system PATH

4. **Run the application**
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Upload PFT Report

- **Option A: Upload File**
  - Click on "Upload File" radio button
  - Upload your PFT report (JPG, JPEG, PNG, or PDF)
  - Click "Extract Values from Document"
  - Review the extracted text and parsed values

- **Option B: Manual Entry**
  - Select "Manual Entry"
  - Enter PFT values directly in the form fields

### 2. Review and Enter Values

- Review automatically extracted values (if uploaded)
- Enter or modify values as needed:
  - **Spirometry:** FEV1, FVC, FEV1/FVC ratio (with % predicted)
  - **Lung Volumes:** TLC, RV (with % predicted)
  - **Diffusion Capacity:** DLCO (with % predicted)
- Click "Save Values and Generate Interpretation"

### 3. View Interpretation

Navigate to the "Analysis & Interpretation" tab to see:
- Summary of entered values
- Step-by-step spirometry analysis
- Lung volume interpretation
- Diffusion capacity assessment
- Final clinical impression with differential diagnoses
- Download option for complete report

### 4. Get AI Review (Optional)

- Navigate to "AI Review" tab
- Enter your Google Gemini API key
- Click "Get AI Review" for AI-powered analysis
- Review additional insights and recommendations

## 🔑 Getting a Gemini API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy and paste into the application when prompted

**Note:** The Gemini API has a free tier with generous limits suitable for personal use.

## 📊 Interpretation Guidelines

The application follows standard pulmonary function test interpretation guidelines:

### Spirometry Patterns

- **Obstructive:** FEV1/FVC < 70%
- **Restrictive:** FVC < 80% predicted with preserved FEV1/FVC ratio
- **Mixed:** Both obstructive and restrictive features
- **Normal:** FEV1/FVC ≥ 70%, FVC ≥ 80% predicted

### Severity Classification (based on FEV1 % predicted)

- **Normal:** ≥ 80%
- **Mild:** 70-79%
- **Moderate:** 60-69%
- **Moderately Severe:** 50-59%
- **Severe:** 35-49%
- **Very Severe:** < 35%

## 🛠️ Technologies Used

- **Streamlit** - Web application framework
- **Pytesseract** - OCR for text extraction
- **Pillow (PIL)** - Image processing
- **pdf2image** - PDF to image conversion
- **Pandas** - Data manipulation and display
- **Google Generative AI (Gemini)** - AI-powered analysis

## 📁 Project Structure

```
pft-analysis-tool/
│
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # Documentation
└── .gitignore           # Git ignore file (create if needed)
```

## 🚢 Deployment

### Deploy on Streamlit Cloud

1. **Push to GitHub**
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Choose `app.py` as the main file
   - Click "Deploy"

3. **Configure Tesseract (Important!)**

Create a file named `packages.txt` in your repository root:
```
tesseract-ocr
poppler-utils
```

This ensures Tesseract OCR and PDF utilities are installed on Streamlit Cloud.

### Deploy on Other Platforms

#### Heroku
Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

Add `packages.txt` with system dependencies.

#### Docker
Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

## ⚠️ Important Notes

- **Medical Disclaimer:** This tool is for educational and informational purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider.

- **Data Privacy:** All processing is done locally/on the server. No PFT data is stored permanently. For production use with real patient data, ensure HIPAA compliance.

- **OCR Accuracy:** OCR results may vary based on image quality. Always review extracted values before generating interpretation.

- **AI Review:** The AI review feature requires an active internet connection and valid Gemini API key. Results should be used as supplementary information only.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 🐛 Known Issues & Troubleshooting

### Tesseract Not Found Error
- **Solution:** Ensure Tesseract is installed and added to system PATH
- **Windows:** Add Tesseract installation directory to PATH environment variable
- **Linux/Mac:** Install via package manager

### PDF Extraction Not Working
- **Solution:** Install poppler-utils
  - Ubuntu: `sudo apt-get install poppler-utils`
  - macOS: `brew install poppler`

### Gemini API Errors
- Verify API key is correct
- Check internet connection
- Ensure API quota is not exceeded

## 📧 Support

For issues, questions, or suggestions, please open an issue on GitHub.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)
- AI analysis by [Google Gemini](https://deepmind.google/technologies/gemini/)

## 📚 References

- American Thoracic Society (ATS) Guidelines for PFT Interpretation
- Global Initiative for Chronic Obstructive Lung Disease (GOLD)
- European Respiratory Society (ERS) Standards

---

**Made with ❤️ for better pulmonary care**
