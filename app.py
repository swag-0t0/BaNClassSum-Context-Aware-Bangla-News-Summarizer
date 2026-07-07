import os
import sys
import time

# Disable safetensors mmap before any model imports (Windows low-RAM systems)
os.environ.setdefault("SAFETENSORS_USE_MMAP", "0")

import streamlit as st

# Project root — used for imports and default model paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Page Config
st.set_page_config(
    page_title="BaNClassSum - From Text Classification to Abstractive Summarization: An End-to-End Pipeline for Bangla News",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Noto+Sans+Bengali:wght@300;400;600;700&display=swap');
    
    /* Main body background & general font */
    .stApp {
        background-color: #0F172A;
        color: #E2E8F0;
        font-family: 'Outfit', 'Noto Sans Bengali', sans-serif;
    }
    
    /* Header Gradient styling */
    .title-container {
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
        background: radial-gradient(circle at top, rgba(99, 102, 241, 0.15) 0%, transparent 70%);
    }
    .title-text {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        background: linear-gradient(135deg, #818CF8 0%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        font-size: 1.1rem;
        color: #94A3B8;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* Custom card styles */
    .glass-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.3);
    }
    
    .glass-card-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #F1F5F9;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Streamlit widgets overrides */
    .stTextArea textarea {
        background-color: #1E293B !important;
        color: #F1F5F9 !important;
        border: 1px solid #334155 !important;
        border-radius: 12px !important;
        padding: 16px !important;
        font-size: 1.05rem !important;
        line-height: 1.6 !important;
    }
    .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.25) !important;
    }
    
    /* Button Styles */
    div.stButton > button {
        background: linear-gradient(135deg, #6366F1 0%, #4F46E5 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        border: none !important;
        width: 100% !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4) !important;
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #3730A3 100%) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px 0 rgba(99, 102, 241, 0.6) !important;
    }
    div.stButton > button:active {
        transform: translateY(0px) !important;
    }
    
    /* Badge styling */
    .badge-container {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .badge-category {
        background: linear-gradient(135deg, #818CF8 0%, #6366F1 100%);
        color: white;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.25);
    }
    .badge-confidence {
        background: rgba(52, 211, 153, 0.1);
        border: 1px solid rgba(52, 211, 153, 0.2);
        color: #34D399;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .badge-mode {
        background: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        padding: 6px 14px;
        border-radius: 9999px;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    /* Footer styling */
    .footer-text {
        text-align: center;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# App Title & Header
st.markdown("""
<div class="title-container">
    <div class="title-text">BaNClassSum</div>
    <div class="subtitle-text">From Text Classification to Abstractive Summarization: An
End-to-End Pipeline for Bangla News</div>
</div>
""", unsafe_allow_html=True)

# Helper functions for Model existence checking
def models_exist(clf_path, sum_path):
    def _valid_model_dir(path):
        return (
            os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "config.json"))
            and os.path.isfile(os.path.join(path, "model.safetensors"))
        )

    return _valid_model_dir(clf_path), _valid_model_dir(sum_path)

# Sidebar Setup
st.sidebar.markdown("### 🛠️ Configuration")

# Default paths to locally saved checkpoints under Models/
DEFAULT_CLF_PATH = os.path.join(PROJECT_ROOT, "Models", "classifier")
DEFAULT_SUM_PATH = os.path.join(PROJECT_ROOT, "Models", "summarizer")

clf_path = st.sidebar.text_input("BERT Classifier Path", value=DEFAULT_CLF_PATH)
sum_path = st.sidebar.text_input("BanglaT5 Summariser Path", value=DEFAULT_SUM_PATH)

clf_exists, sum_exists = models_exist(clf_path, sum_path)

# Determine Mode
if not (clf_exists and sum_exists):
    st.sidebar.warning("⚠️ Local trained model paths not found.")
    app_mode = st.sidebar.radio("Execution Mode", ["Demo / Mock Mode", "Pipeline Mode (Attempt load)"], index=0)
else:
    st.sidebar.success("✅ Trained model checkpoints detected.")
    app_mode = st.sidebar.radio("Execution Mode", ["Pipeline Mode", "Demo / Mock Mode"], index=0)

# Summarization parameters in Sidebar
st.sidebar.markdown("### ⚙️ Generation Parameters")
max_len = st.sidebar.slider("Max Summary Length", min_value=50, max_value=250, value=128, step=10)
min_len = st.sidebar.slider("Min Summary Length", min_value=10, max_value=100, value=30, step=5)

# Load pipeline function
@st.cache_resource(show_spinner="Loading BERT classifier and BanglaT5 summariser...")
def load_pipeline(c_path, s_path):
    try:
        import Pipeline.unified_pipeline as up

        up.CLASSIFIER_PATH = os.path.abspath(c_path)
        up.SUMMARISER_PATH = os.path.abspath(s_path)

        return up.BanglaNewsPipeline(), None
    except Exception as e:
        return None, str(e)

# Run Mock model for classification and summarisation
def run_mock_pipeline(text, max_length, min_length):
    # Simulating category classification
    text_lower = text.lower()
    
    # Simple rule-based mock classifier
    categories = {
        "খেলাধুলা (Sports)": ["খেলা", "ক্রিকেট", "ফুটবল", "রান", "উইকেট", "ম্যাচ", "স্টেডিয়াম", "প্লেয়ার", "দল", "অধিনায়ক"],
        "রাজনীতি (Politics)": ["রাজনীতি", "নির্বাচন", "দল", "বিএনপি", "আওয়ামী", "লীগ", "সংসদ", "মন্ত্রী", "প্রধানমন্ত্রী", "ভোট"],
        "অর্থনীতি (Economy)": ["অর্থনীতি", "টাকা", "বজেট", "ব্যাংক", "শেয়ার", "বাজার", "বিনিয়োগ", "রাজস্ব", "মুদ্রাস্ফীতি"],
        "আন্তর্জাতিক (International)": ["আন্তর্জাতিক", "যুক্তরাষ্ট্র", "ভারত", "চীন", "ইউক্রেন", "রাশিয়া", "বিশ্ব", "সীমান্ত"],
        "জাতীয় (National)": ["জাতীয়", "বাংলাদেশ", "ঢাকা", "পুলিশ", "আদালত", "মামলা", "আইন", "উদ্বোধন", "দুর্ঘটনা"]
    }
    
    detected_cat = "জাতীয় (National)"
    max_matches = 0
    
    for cat, keywords in categories.items():
        matches = sum(1 for kw in keywords if kw in text_lower)
        if matches > max_matches:
            max_matches = matches
            detected_cat = cat
            
    confidence = 0.85 + (min(max_matches, 5) * 0.02)
    
    # Simple extractive mock summarizer — respects max/min length (~4 chars/token for Bangla)
    char_min = min_length * 4
    char_max = max_length * 4
    sentences = [s.strip() for s in text.split("।") if s.strip()]
    parts = []

    for sentence in sentences:
        trial = ("। ".join(parts + [sentence]) + "।") if parts else (sentence + "।")
        if parts and len(trial) > char_max:
            break
        parts.append(sentence)

    if parts and len("। ".join(parts) + "।") < char_min:
        for sentence in sentences[len(parts):]:
            trial = "। ".join(parts + [sentence]) + "।"
            if len(trial) > char_max:
                break
            parts.append(sentence)
            if len(trial) >= char_min:
                break

    summary = ("। ".join(parts) + "।") if parts else text
    if len(summary) > char_max:
        summary = summary[:char_max].rsplit(" ", 1)[0] + "…"
        
    return {
        "category": detected_cat,
        "confidence": confidence,
        "summary": summary,
        "max_length": max_length,
        "min_length": min_length,
    }

# Main Layout
col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📝 Input News Content</div>', unsafe_allow_html=True)
    
    default_text = (
        "ডিজিটাল নিরাপত্তা আইনে গ্রেফতার বাংলা ট্রিবিউনের খুলনা প্রতিনিধি হেদায়েত হোসেন মোল্লাকে "
        "জিজ্ঞাসাবাদের জন্য তিনদিনের রিমান্ড মঞ্জুর করেছে আদালত। পুলিশের আবেদনের প্রেক্ষিতে খুলনার "
        "সিনিয়র জুডিশিয়াল ম্যাজিস্ট্রেট আদালত এই রিমান্ড মঞ্জুর করেন। এর আগে নির্বাচন সংক্রান্ত "
        "ভুল সংবাদ পরিবেশনের অভিযোগে তাকে আটক করা হয়েছিল।"
    )
    
    input_text = st.text_area(
        label="Paste Bangla News article here:",
        value=default_text,
        height=320,
        placeholder="নিউজ আর্টিকেল এখানে পেস্ট করুন...",
        label_visibility="collapsed"
    )
    
    summarise_button = st.button("✨ Summarise & Classify")
    st.markdown('</div>', unsafe_allow_html=True)

with col_output:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-card-header">📊 Result Panel</div>', unsafe_allow_html=True)
    
    if summarise_button:
        if not input_text.strip():
            st.error("Please enter some text to process.")
        else:
            with st.spinner("Processing news article... Please wait."):
                # Add a minor delay for smoother UI feedback transition
                time.sleep(0.8)
                
                result = None
                error_msg = None
                mode_used = ""
                
                if "Pipeline Mode" in app_mode:
                    # Attempt to run real pipeline
                    pipeline, err = load_pipeline(clf_path, sum_path)
                    if pipeline is not None:
                        try:
                            # We override run to support sliders
                            category, confidence = pipeline.classify(input_text)
                            summary = pipeline.summarise(
                                input_text, 
                                category, 
                                max_length=max_len, 
                                min_length=min_len
                            )
                            result = {
                                "category": category,
                                "confidence": confidence,
                                "summary": summary,
                                "max_length": max_len,
                                "min_length": min_len,
                            }
                            mode_used = "Pipeline Mode"
                        except Exception as e:
                            error_msg = f"Failed executing pipeline: {str(e)}"
                    else:
                        error_msg = f"Failed to load pipeline resources: {err}"
                        
                # Fallback to Demo Mode if requested, or if pipeline run encountered an error
                if result is None:
                    if "Pipeline Mode" in app_mode:
                        st.info(f"ℹ️ Falling back to Demo Mode. Error Details: {error_msg}")
                    result = run_mock_pipeline(input_text, max_len, min_len)
                    mode_used = "Demo / Mock Mode"
                
                # Render metadata badges
                max_used = result.get("max_length", max_len)
                min_used = result.get("min_length", min_len)
                st.markdown(f"""
                <div class="badge-container">
                    <span class="badge-mode">⚙️ {mode_used}</span>
                    <span class="badge-category">📁 {result['category']}</span>
                    <span class="badge-confidence">🎯 Confidence: {result['confidence']:.2%}</span>
                    <span class="badge-mode">📏 Max: {max_used} · Min: {min_used}</span>
                </div>
                """, unsafe_allow_html=True)
                
                # Show summary
                st.markdown("##### 📌 Summarised Text:")
                st.markdown(f'<div class="summary-text">{result["summary"]}</div>', unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Download option
                st.download_button(
                    label="💾 Download Summary",
                    data=result["summary"],
                    file_name="summary.txt",
                    mime="text/plain"
                )
    else:
        st.info("👈 Enter news content on the left and click 'Summarise & Classify' to see results.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-text">
    BaNClassSum | Developed as a Context-Aware Bangla News Summarizer & Classifier
</div>
""", unsafe_allow_html=True)
