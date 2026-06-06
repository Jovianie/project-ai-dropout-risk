import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu
import time

# Page configuration
st.set_page_config(
    page_title="EduTrack | Prediksi Dropout Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS - Detailed styling
st.markdown("""
<style>
/* ==================== GOOGLE FONTS ==================== */
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300;1,400;1,500;1,600;1,700&family=Inter:ital,wght@0,100;0,200;0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,100;1,200;1,300;1,400;1,500;1,600;1,700;1,800;1,900&display=swap');

/* ==================== ROOT VARIABLES ==================== */
:root {
    --primary: #8B9D9C;
    --primary-light: #A5B8B7;
    --primary-dark: #6B7F7E;
    --secondary: #D4C5B3;
    --secondary-light: #E8DDD1;
    --secondary-dark: #BFA88C;
    --accent: #C17A6B;
    --accent-light: #D99B8C;
    --bg-main: #F8F6F2;
    --bg-card: #FFFFFF;
    --bg-sidebar: #F0EDE8;
    --text-dark: #2C3E35;
    --text-muted: #6B7F7E;
    --text-light: #9BA89F;
    --success: #8FBC8F;
    --warning: #D4A373;
    --danger: #C17A6B;
    --border: #E8DDD1;
    --shadow-sm: 0 2px 8px rgba(0,0,0,0.04);
    --shadow-md: 0 8px 24px rgba(0,0,0,0.06);
    --shadow-lg: 0 16px 40px rgba(0,0,0,0.08);
}

/* ==================== BASE STYLES ==================== */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Inter', sans-serif;
    background-color: var(--bg-main);
    color: var(--text-dark);
}

/* ==================== TYPOGRAPHY ==================== */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Cormorant Garamond', serif;
    font-weight: 600;
    letter-spacing: -0.02em;
}

h1 {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem;
}

h2 {
    font-size: 2rem;
    font-weight: 600;
    color: var(--primary-dark);
    margin-bottom: 1.5rem;
    border-left: 4px solid var(--accent);
    padding-left: 1rem;
}

h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: 1rem;
}

h4 {
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--primary-dark);
    margin-bottom: 0.75rem;
}

.subtitle {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.25rem;
    font-style: italic;
    color: var(--text-muted);
    margin-bottom: 2rem;
}

/* ==================== HEADER SECTION ==================== */
.main-header {
    background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-sidebar) 100%);
    padding: 2rem 2rem 1.5rem 2rem;
    border-radius: 24px;
    margin-bottom: 2rem;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 1rem;
}

.header-logo {
    display: flex;
    align-items: center;
    gap: 1rem;
}

.logo-icon {
    font-size: 2.5rem;
}

.header-tagline {
    font-family: 'Cormorant Garamond', serif;
    font-style: italic;
    font-size: 1rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* ==================== CARD STYLES ==================== */
.stats-card {
    background: var(--bg-card);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
    border: 1px solid var(--border);
    transition: all 0.3s ease;
}

.stats-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-md);
}

.stats-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--primary-dark);
}

.stats-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prediction-card {
    background: var(--bg-card);
    border-radius: 28px;
    padding: 2rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, var(--accent), var(--primary-light));
}

.prediction-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 500;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}

.prediction-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3rem;
    font-weight: 700;
    margin: 0.5rem 0;
}

.prediction-value.high-risk {
    color: var(--danger);
}

.prediction-value.low-risk {
    color: var(--success);
}

.prediction-value.medium-risk {
    color: var(--warning);
}

.prediction-probability {
    font-size: 0.9rem;
    color: var(--text-muted);
    margin-top: 0.5rem;
}

/* ==================== FORM STYLES ==================== */
.form-container {
    background: var(--bg-card);
    border-radius: 28px;
    padding: 2rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border);
}

.form-section {
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}

.form-section:last-child {
    border-bottom: none;
    margin-bottom: 0;
    padding-bottom: 0;
}

.form-section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.2rem;
    font-weight: 600;
    color: var(--primary-dark);
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.section-icon {
    font-size: 1.2rem;
}

/* Input field styling */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div > select,
.stSlider > div > div {
    background-color: var(--bg-sidebar);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.5rem 1rem;
    font-family: 'Inter', sans-serif;
    transition: all 0.3s ease;
}

.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 2px rgba(193, 122, 107, 0.1);
}

/* Label styling */
.stTextInput > label,
.stNumberInput > label,
.stSelectbox > label,
.stSlider > label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    font-weight: 500;
    color: var(--text-dark);
    margin-bottom: 0.25rem;
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    border: none;
    border-radius: 40px;
    padding: 0.75rem 2rem;
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    font-size: 1rem;
    transition: all 0.3s ease;
    width: 100%;
    cursor: pointer;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(107, 127, 126, 0.2);
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
}

.stButton > button:active {
    transform: translateY(0);
}

/* Reset button */
.reset-button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text-muted) !important;
    box-shadow: none !important;
}

.reset-button:hover {
    background: var(--bg-sidebar) !important;
    transform: none !important;
}

/* ==================== INFO BOXES ==================== */
.info-box {
    background: var(--bg-sidebar);
    border-radius: 20px;
    padding: 1.5rem;
    margin-top: 1.5rem;
    border-left: 3px solid var(--accent);
}

.info-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--primary-dark);
    margin-bottom: 0.5rem;
}

.info-text {
    font-size: 0.85rem;
    color: var(--text-muted);
    line-height: 1.5;
}

.warning-box {
    background: rgba(212, 197, 179, 0.3);
    border-radius: 16px;
    padding: 1rem;
    margin-bottom: 1rem;
    border-left: 3px solid var(--warning);
}

.warning-text {
    font-size: 0.85rem;
    color: var(--text-dark);
}

/* ==================== METRIC CARDS ==================== */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.metric-item {
    background: var(--bg-sidebar);
    border-radius: 16px;
    padding: 1rem;
    text-align: center;
}

.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-dark);
}

.metric-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    margin-top: 0.25rem;
}

/* ==================== SLIDER CUSTOMIZATION ==================== */
.stSlider > div > div > div {
    background-color: var(--accent-light);
}

.stSlider > div > div > div > div {
    background-color: var(--accent);
}

/* ==================== TAB STYLES ==================== */
.stTabs [data-baseweb="tab-list"] {
    gap: 2rem;
    background-color: transparent;
    padding: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif;
    font-size: 0.9rem;
    font-weight: 500;
    color: var(--text-muted);
    padding: 0.5rem 0;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: var(--primary-dark);
    border-bottom: 2px solid var(--accent);
}

/* ==================== EXPANDER STYLES ==================== */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif;
    font-weight: 500;
    color: var(--primary-dark);
    background-color: var(--bg-sidebar);
    border-radius: 12px;
}

.streamlit-expanderContent {
    background-color: var(--bg-card);
    border-radius: 12px;
    padding: 1rem;
}

/* ==================== ALERT STYLES ==================== */
.success-alert {
    background: rgba(143, 188, 143, 0.1);
    border-left: 4px solid var(--success);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

.warning-alert {
    background: rgba(212, 163, 115, 0.1);
    border-left: 4px solid var(--warning);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

.danger-alert {
    background: rgba(193, 122, 107, 0.1);
    border-left: 4px solid var(--danger);
    border-radius: 12px;
    padding: 1rem;
    margin: 1rem 0;
}

/* ==================== FOOTER ==================== */
.footer {
    text-align: center;
    padding: 2rem;
    margin-top: 3rem;
    border-top: 1px solid var(--border);
    color: var(--text-light);
    font-size: 0.8rem;
}

/* ==================== ANIMATIONS ==================== */
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.fade-in {
    animation: fadeIn 0.5s ease-out;
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
    }
    50% {
        opacity: 0.5;
    }
}

.pulse {
    animation: pulse 2s ease-in-out infinite;
}

/* ==================== RESPONSIVE DESIGN ==================== */
@media (max-width: 768px) {
    h1 {
        font-size: 2rem;
    }
    
    h2 {
        font-size: 1.5rem;
    }
    
    .stats-card {
        padding: 1rem;
    }
    
    .form-container {
        padding: 1.5rem;
    }
    
    .metric-grid {
        grid-template-columns: repeat(2, 1fr);
    }
}

@media (max-width: 480px) {
    .metric-grid {
        grid-template-columns: 1fr;
    }
    
    .header-content {
        flex-direction: column;
        text-align: center;
    }
}

/* ==================== SCROLLBAR STYLING ==================== */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-sidebar);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-light);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary);
}

/* ==================== LOADING SPINNER ==================== */
.loading-spinner {
    display: inline-block;
    width: 40px;
    height: 40px;
    border: 3px solid var(--border);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
}

@keyframes spin {
    to {
        transform: rotate(360deg);
    }
}

/* ==================== TOOLTIP ==================== */
[data-tooltip] {
    position: relative;
    cursor: help;
}

[data-tooltip]:before {
    content: attr(data-tooltip);
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    padding: 0.5rem 0.75rem;
    background: var(--text-dark);
    color: white;
    font-size: 0.75rem;
    border-radius: 8px;
    white-space: nowrap;
    opacity: 0;
    visibility: hidden;
    transition: all 0.3s ease;
    z-index: 1000;
}

[data-tooltip]:hover:before {
    opacity: 1;
    visibility: visible;
}

/* ==================== DARK MODE SUPPORT ==================== */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: #1a1f1e;
        --bg-card: #252d2b;
        --bg-sidebar: #1f2624;
        --text-dark: #E8DDD1;
        --text-muted: #9BA89F;
        --border: #3a4542;
    }
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load the trained model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.warning(f"Model file not found. Using built-in prediction logic. Error: {e}")
        return None

model = load_model()

# Prediction function
def predict_dropout(input_data, model):
    """
    Prediction based on trained Logistic Regression model or fallback logic
    """
    if model is not None:
        try:
            # Prepare input for model
            input_df = pd.DataFrame([input_data])
            probability = model.predict_proba(input_df)[0][1]
            return probability
        except Exception as e:
            st.warning(f"Model prediction error, using fallback. Error: {e}")
    
    # Fallback prediction based on trained coefficients
    coefficients = {
        'GPA': -1.262264,
        'Stress_Index': 0.309133,
        'Assignment_Delay_Days': 0.248695,
        'Attendance_Rate': -0.239672,
        'Study_Hours_per_Day': -0.027219,
        'Travel_Time_Minutes': 0.125791,
        'Part_Time_Job_Yes': 0.022302,
        'Part_Time_Job_No': -0.177313,
        'Internet_Access_Yes': -0.137767,
        'Scholarship_Yes': -0.113611,
        'Scholarship_No': -0.041399,
        'Semester_Year_1': -0.117075,
        'Semester_Year_2': 0.027091,
        'Semester_Year_3': -0.018158,
        'Semester_Year_4': -0.046868
    }
    
    intercept = -0.5
    
    log_odds = intercept
    log_odds += coefficients['GPA'] * input_data['GPA']
    log_odds += coefficients['Stress_Index'] * input_data['Stress_Index']
    log_odds += coefficients['Assignment_Delay_Days'] * input_data['Assignment_Delay_Days']
    
    attendance_normalized = (input_data['Attendance_Rate'] - 81.74) / 8.22
    log_odds += coefficients['Attendance_Rate'] * attendance_normalized
    
    study_hours_normalized = (input_data['Study_Hours_per_Day'] - 4.01) / 1.30
    log_odds += coefficients['Study_Hours_per_Day'] * study_hours_normalized
    
    travel_normalized = (input_data['Travel_Time_Minutes'] - 30.18) / 11.92
    log_odds += coefficients['Travel_Time_Minutes'] * travel_normalized
    
    if input_data['Part_Time_Job'] == 'Yes':
        log_odds += coefficients['Part_Time_Job_Yes']
    else:
        log_odds += coefficients['Part_Time_Job_No']
    
    if input_data['Internet_Access'] == 'Yes':
        log_odds += coefficients['Internet_Access_Yes']
    
    if input_data['Scholarship'] == 'Yes':
        log_odds += coefficients['Scholarship_Yes']
    else:
        log_odds += coefficients['Scholarship_No']
    
    semester_key = f"Semester_{input_data['Semester'].replace(' ', '_')}"
    log_odds += coefficients.get(semester_key, 0)
    
    probability = 1 / (1 + np.exp(-log_odds))
    return probability

# Header section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("""
    <div class="main-header fade-in">
        <div class="header-content">
            <div class="header-logo">
                <div class="logo-icon">🎓</div>
                <div>
                    <h1>EduTrack</h1>
                    <div class="header-tagline">Predict • Prevent • Progress</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card fade-in" style="margin-top: 0.5rem;">
        <div class="stats-number">87.5%</div>
        <div class="stats-label">Model Accuracy</div>
        <div class="info-text" style="margin-top: 0.5rem;">Logistic Regression</div>
    </div>
    """, unsafe_allow_html=True)

# Navigation
selected = option_menu(
    menu_title=None,
    options=["📋 Prediksi", "📊 Statistik", "📘 Panduan"],
    icons=["clipboard-data", "bar-chart-steps", "book"],
    orientation="horizontal",
    default_index=0,
    styles={
        "container": {"padding": "0!important", "background-color": "transparent"},
        "icon": {"color": "var(--accent)", "font-size": "1rem"},
        "nav-link": {
            "font-family": "Inter, sans-serif",
            "font-size": "0.9rem",
            "color": "var(--text-muted)",
            "text-align": "center",
            "margin": "0 0.5rem",
            "padding": "0.5rem 1rem",
            "border-radius": "40px",
        },
        "nav-link-selected": {
            "background": "linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%)",
            "color": "white",
        },
    },
)

if selected == "📋 Prediksi":
    left_col, right_col = st.columns([3, 2])
    
    with left_col:
        st.markdown('<div class="form-container fade-in">', unsafe_allow_html=True)
        
        # Student Information Section
        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">
                <span class="section-icon">👤</span>
                <span>Informasi Mahasiswa</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            semester = st.selectbox(
                "Semester",
                ["Year 1", "Year 2", "Year 3", "Year 4"],
                help="Tahun studi mahasiswa saat ini"
            )
        
        with col_b:
            gender = st.selectbox(
                "Jenis Kelamin",
                ["Laki-laki", "Perempuan"],
                help="Jenis kelamin mahasiswa"
            )
        
        # Academic Section
        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">
                <span class="section-icon">📚</span>
                <span>Data Akademik</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_c, col_d = st.columns(2)
        with col_c:
            gpa = st.number_input(
                "IPK (0-4.0)",
                min_value=0.0,
                max_value=4.0,
                value=2.5,
                step=0.01,
                format="%.2f",
                help="Indeks Prestasi Kumulatif"
            )
            
            study_hours = st.slider(
                "Jam Belajar per Hari",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
                help="Rata-rata waktu belajar per hari"
            )
            
            assignment_delay = st.slider(
                "Keterlambatan Tugas (hari)",
                min_value=0,
                max_value=10,
                value=2,
                help="Rata-rata keterlambatan pengumpulan tugas"
            )
        
        with col_d:
            attendance = st.slider(
                "Tingkat Kehadiran (%)",
                min_value=0,
                max_value=100,
                value=85,
                help="Persentase kehadiran dalam perkuliahan"
            )
            
            stress_index = st.slider(
                "Tingkat Stres (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                help="Tingkat stres yang dirasakan"
            )
            
            travel_time = st.number_input(
                "Waktu Perjalanan (menit)",
                min_value=0,
                max_value=180,
                value=30,
                step=5,
                help="Waktu tempuh ke kampus"
            )
        
        # Socioeconomic Section
        st.markdown("""
        <div class="form-section">
            <div class="form-section-title">
                <span class="section-icon">🏠</span>
                <span>Data Sosial Ekonomi</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col_e, col_f, col_g = st.columns(3)
        with col_e:
            internet_access = st.selectbox(
                "Akses Internet",
                ["Ya", "Tidak"],
                help="Ketersediaan akses internet"
            )
        
        with col_f:
            part_time_job = st.selectbox(
                "Pekerjaan Paruh Waktu",
                ["Ya", "Tidak"],
                help="Apakah memiliki pekerjaan paruh waktu"
            )
        
        with col_g:
            scholarship = st.selectbox(
                "Beasiswa",
                ["Ya", "Tidak"],
                help="Apakah menerima beasiswa"
            )
        
        # Map Indonesian values to model values
        internet_map = {"Ya": "Yes", "Tidak": "No"}
        part_time_map = {"Ya": "Yes", "Tidak": "No"}
        scholarship_map = {"Ya": "Yes", "Tidak": "No"}
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with right_col:
        st.markdown('<div class="form-container fade-in">', unsafe_allow_html=True)
        st.markdown('<div class="form-section-title"><span class="section-icon">🔮</span><span>Hasil Prediksi</span></div>', unsafe_allow_html=True)
        
        predict_clicked = st.button("🚀 Prediksi Risiko Dropout", use_container_width=True)
        
        if predict_clicked:
            with st.spinner("Menganalisis data mahasiswa..."):
                time.sleep(0.5)
                
                input_data = {
                    'GPA': gpa,
                    'Stress_Index': stress_index,
                    'Assignment_Delay_Days': assignment_delay,
                    'Attendance_Rate': attendance,
                    'Study_Hours_per_Day': study_hours,
                    'Travel_Time_Minutes': travel_time,
                    'Part_Time_Job': part_time_map[part_time_job],
                    'Internet_Access': internet_map[internet_access],
                    'Scholarship': scholarship_map[scholarship],
                    'Semester': semester
                }
                
                probability = predict_dropout(input_data, model)
                
                if probability >= 0.6:
                    risk_level = "Tinggi"
                    risk_class = "high-risk"
                    risk_icon = "⚠️"
                    recommendation = "Rekomendasi: Segera konsultasi dengan dosen pembimbing dan program studi untuk intervensi akademik intensif."
                elif probability >= 0.4:
                    risk_level = "Sedang"
                    risk_class = "medium-risk"
                    risk_icon = "⚡"
                    recommendation = "Rekomendasi: Tingkatkan partisipasi dalam perkuliahan dan manfaatkan layanan konseling akademik yang tersedia."
                else:
                    risk_level = "Rendah"
                    risk_class = "low-risk"
                    risk_icon = "✅"
                    recommendation = "Rekomendasi: Pertahankan prestasi akademik yang baik dan terus aktif dalam kegiatan perkuliahan."
                
                result_label = f"{risk_icon} Risiko Dropout: {risk_level}"
                
                st.markdown(f"""
                <div class="prediction-card fade-in">
                    <div class="prediction-label">{result_label}</div>
                    <div class="prediction-value {risk_class}">{probability:.1%}</div>
                    <div class="prediction-probability">Probabilitas dropout</div>
                </div>
                """, unsafe_allow_html=True)
                
                if risk_level == "Tinggi":
                    st.markdown(f"""
                    <div class="danger-alert">
                        <strong>⚠️ Perhatian!</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                elif risk_level == "Sedang":
                    st.markdown(f"""
                    <div class="warning-alert">
                        <strong>⚡ Perlu Perhatian</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-alert">
                        <strong>✅ Dalam Jalur yang Baik</strong><br>
                        {recommendation}
                    </div>
                    """, unsafe_allow_html=True)
                
                history_entry = {
                    'GPA': gpa,
                    'Semester': semester,
                    'Attendance': attendance,
                    'Study_Hours': study_hours,
                    'Probability': probability,
                    'Risk': risk_level
                }
                st.session_state.prediction_history.append(history_entry)
                
                if len(st.session_state.prediction_history) > 10:
                    st.session_state.prediction_history = st.session_state.prediction_history[-10:]
        
        else:
            st.markdown("""
            <div class="info-box">
                <div class="info-title">✨ Siap Memprediksi?</div>
                <div class="info-text">
                    Lengkapi data mahasiswa di sebelah kiri, lalu klik tombol "Prediksi Risiko Dropout"<br>
                    untuk mendapatkan analisis risiko kelulusan.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-box">
                <div class="info-title">📊 Faktor Kunci</div>
                <div class="info-text">
                    Berdasarkan model prediksi, faktor-faktor berikut memiliki pengaruh signifikan terhadap risiko dropout:
                    <ul style="margin-top: 0.5rem; margin-left: 1rem;">
                        <li>IPK (Indeks Prestasi Kumulatif)</li>
                        <li>Tingkat stres akademik</li>
                        <li>Frekuensi keterlambatan tugas</li>
                        <li>Tingkat kehadiran</li>
                        <li>Durasi belajar harian</li>
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif selected == "📊 Statistik":
    st.markdown('<div class="form-container fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title"><span class="section-icon">📈</span><span>Statistik Prediksi</span></div>', unsafe_allow_html=True)
    
    if len(st.session_state.prediction_history) > 0:
        history_df = pd.DataFrame(st.session_state.prediction_history)
        
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{len(history_df)}</div>
                <div class="stats-label">Total Prediksi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            high_risk_count = len(history_df[history_df['Risk'] == 'Tinggi'])
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number" style="color: var(--danger);">{high_risk_count}</div>
                <div class="stats-label">Risiko Tinggi</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            avg_prob = history_df['Probability'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{avg_prob:.1%}</div>
                <div class="stats-label">Rata-rata Probabilitas</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            avg_gpa = history_df['GPA'].mean()
            st.markdown(f"""
            <div class="stats-card">
                <div class="stats-number">{avg_gpa:.2f}</div>
                <div class="stats-label">Rata-rata IPK</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<h4>📊 Riwayat Prediksi</h4>", unsafe_allow_html=True)
        
        history_df_display = history_df.copy()
        history_df_display['Index'] = range(1, len(history_df_display) + 1)
        
        fig = go.Figure()
        
        colors = ['#C17A6B' if x >= 0.6 else '#D4A373' if x >= 0.4 else '#8FBC8F' for x in history_df_display['Probability']]
        
        fig.add_trace(go.Bar(
            x=history_df_display['Index'],
            y=history_df_display['Probability'],
            marker_color=colors,
            text=[f'{p:.1%}' for p in history_df_display['Probability']],
            textposition='outside',
            name='Probabilitas Dropout'
        ))
        
        fig.update_layout(
            title="Tren Probabilitas Dropout",
            xaxis_title="Urutan Prediksi",
            yaxis_title="Probabilitas Dropout",
            yaxis_tickformat='.0%',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="Inter, sans-serif", color="#2C3E35"),
            height=400,
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<h4>📋 Detail Riwayat</h4>", unsafe_allow_html=True)
        
        display_df = history_df.copy()
        display_df['Probability'] = display_df['Probability'].apply(lambda x: f"{x:.1%}")
        display_df = display_df[['Semester', 'GPA', 'Attendance', 'Study_Hours', 'Probability', 'Risk']]
        display_df.columns = ['Semester', 'IPK', 'Kehadiran (%)', 'Jam Belajar', 'Probabilitas', 'Risiko']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "IPK": st.column_config.NumberColumn(format="%.2f"),
                "Kehadiran (%)": st.column_config.NumberColumn(format="%.0f%%"),
                "Probabilitas": st.column_config.TextColumn(),
                "Risiko": st.column_config.TextColumn(),
            }
        )
        
        if st.button("🗑️ Hapus Riwayat", use_container_width=True):
            st.session_state.prediction_history = []
            st.rerun()
    
    else:
        st.info("Belum ada data prediksi. Silakan lakukan prediksi terlebih dahulu pada menu 'Prediksi'.")
    
    st.markdown('</div>', unsafe_allow_html=True)

elif selected == "📘 Panduan":
    st.markdown('<div class="form-container fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title"><span class="section-icon">📘</span><span>Panduan Penggunaan</span></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-bottom: 1.5rem;">
        <div class="info-title">🎯 Cara Menggunakan Aplikasi</div>
        <div class="info-text">
            <ol style="margin-top: 0.5rem; margin-left: 1rem;">
                <li>Lengkapi data mahasiswa pada formulir di halaman <strong>Prediksi</strong></li>
                <li>Pastikan semua data diisi dengan akurat sesuai kondisi mahasiswa</li>
                <li>Klik tombol <strong>"Prediksi Risiko Dropout"</strong> untuk memulai analisis</li>
                <li>Sistem akan menampilkan probabilitas risiko dropout beserta rekomendasi</li>
                <li>Lihat riwayat prediksi dan statistik di halaman <strong>Statistik</strong></li>
            </ol>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-bottom: 1.5rem;">
        <div class="info-title">🤖 Tentang Model Prediksi</div>
        <div class="info-text">
            Model prediksi ini dibangun menggunakan algoritma <strong>Logistic Regression</strong> 
            dengan tingkat akurasi mencapai <strong>87.5%</strong>. Model telah dilatih menggunakan data 
            historis mahasiswa dengan berbagai faktor yang mempengaruhi risiko dropout.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <div class="info-title">📊 Faktor yang Dianalisis</div>
        <div class="info-text">
            <ul style="margin-top: 0.5rem; margin-left: 1rem;">
                <li><strong>IPK (GPA)</strong> - Indikator utama performa akademik</li>
                <li><strong>Tingkat Stres</strong> - Pengaruh kesehatan mental terhadap studi</li>
                <li><strong>Keterlambatan Tugas</strong> - Indikator kedisiplinan akademik</li>
                <li><strong>Tingkat Kehadiran</strong> - Partisipasi dalam perkuliahan</li>
                <li><strong>Jam Belajar</strong> - Intensitas belajar mandiri</li>
                <li><strong>Waktu Perjalanan</strong> - Aksesibilitas ke kampus</li>
                <li><strong>Status Pekerjaan</strong> - Beban non-akademik mahasiswa</li>
                <li><strong>Akses Internet</strong> - Dukungan fasilitas belajar</li>
                <li><strong>Status Beasiswa</strong> - Dukungan finansial</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box" style="margin-top: 1.5rem;">
        <div class="info-title">💡 Tips Pencegahan Dropout</div>
        <div class="info-text">
            <ul style="margin-top: 0.5rem; margin-left: 1rem;">
                <li>Jaga konsistensi kehadiran dalam perkuliahan (minimal 80%)</li>
                <li>Kelola waktu belajar dengan baik (minimal 4-5 jam per hari)</li>
                <li>Jangan menunda pengumpulan tugas</li>
                <li>Manfaatkan layanan konseling jika mengalami stres akademik</li>
                <li>Bangun relasi positif dengan dosen dan teman sekelas</li>
                <li>Ikuti kegiatan pengembangan soft skills di kampus</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <div>© 2024 EduTrack — Prediksi Risiko Dropout Mahasiswa</div>
    <div style="margin-top: 0.5rem;">Dibangun dengan Logistic Regression | Akurasi Model 87.5%</div>
</div>
""", unsafe_allow_html=True)
