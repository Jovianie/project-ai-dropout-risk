# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

# Page config
st.set_page_config(
    page_title="Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Color palette from image
COLORS = {
    'licorice': '#1C150F',
    'dark_moss': '#5C6844',
    'dutch_white': '#D7CEAF',
    'ash_gray': '#B6CCCA',
    'slate_gray': '#67898C',
    'white': '#FFFFFF',
    'light_bg': '#F5F2EB'
}

# CSS with serif fonts and color palette
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');
    
    * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }}
    
    .stApp {{
        background-color: {COLORS['light_bg']};
    }}
    
    .main-header {{
        text-align: center;
        padding: 2rem 0 1rem 0;
        margin-bottom: 1.5rem;
    }}
    
    h1 {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 3rem;
        font-weight: 600;
        color: {COLORS['licorice']};
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }}
    
    .subhead {{
        font-family: 'Cormorant Garamond', serif;
        font-style: italic;
        font-size: 1rem;
        color: {COLORS['slate_gray']};
        letter-spacing: 0.3px;
    }}
    
    h2, h3 {{
        font-family: 'Cormorant Garamond', serif;
        font-weight: 500;
        color: {COLORS['licorice']};
    }}
    
    .card {{
        background: {COLORS['white']};
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid {COLORS['dutch_white']};
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }}
    
    .result-card {{
        background: {COLORS['white']};
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid {COLORS['dutch_white']};
    }}
    
    .result-value {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 4rem;
        font-weight: 700;
        margin: 1rem 0;
    }}
    
    .risk-high {{ color: #c0392b; }}
    .risk-medium {{ color: #e67e22; }}
    .risk-low {{ color: {COLORS['dark_moss']}; }}
    
    .result-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: {COLORS['slate_gray']};
    }}
    
    .recommendation {{
        background: {COLORS['light_bg']};
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: {COLORS['licorice']};
        border-left: 3px solid {COLORS['slate_gray']};
        text-align: left;
    }}
    
    .stat-card {{
        background: {COLORS['white']};
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid {COLORS['dutch_white']};
    }}
    
    .stat-number {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.75rem;
        font-weight: 600;
        color: {COLORS['licorice']};
    }}
    
    .stat-label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: {COLORS['slate_gray']};
        letter-spacing: 0.5px;
    }}
    
    hr {{
        margin: 1.5rem 0;
        border: none;
        border-top: 1px solid {COLORS['dutch_white']};
    }}
    
    .footer {{
        text-align: center;
        padding: 2rem;
        font-family: 'Inter', sans-serif;
        font-size: 0.7rem;
        color: {COLORS['slate_gray']};
        border-top: 1px solid {COLORS['dutch_white']};
        margin-top: 2rem;
    }}
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {{
        border-radius: 8px;
        border: 1px solid {COLORS['dutch_white']};
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
    }}
    
    .stTextInput > label,
    .stNumberInput > label,
    .stSelectbox > label,
    .stSlider > label {{
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 500;
        color: {COLORS['licorice']};
    }}
    
    .stButton > button {{
        background: {COLORS['dark_moss']};
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.85rem;
        width: 100%;
        transition: all 0.2s;
    }}
    
    .stButton > button:hover {{
        background: {COLORS['slate_gray']};
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 2rem;
        border-bottom: 1px solid {COLORS['dutch_white']};
    }}
    
    .stTabs [data-baseweb="tab"] {{
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: {COLORS['slate_gray']};
        padding: 0.5rem 0;
    }}
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {COLORS['licorice']};
        border-bottom: 2px solid {COLORS['dark_moss']};
    }}
    
    /* Slider */
    .stSlider > div > div > div {{
        background-color: {COLORS['ash_gray']};
    }}
    
    .stSlider > div > div > div > div {{
        background-color: {COLORS['dark_moss']};
    }}
    
    /* Metric */
    [data-testid="stMetricValue"] {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem;
        color: {COLORS['licorice']};
    }}
    
    /* Divider */
    .custom-divider {{
        height: 1px;
        background: {COLORS['dutch_white']};
        margin: 1rem 0;
    }}
    
    .section-title {{
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.25rem;
        font-weight: 500;
        color: {COLORS['licorice']};
        margin-bottom: 1rem;
        border-left: 3px solid {COLORS['dark_moss']};
        padding-left: 0.75rem;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load model
@st.cache_resource
def load_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model, True
    except Exception as e:
        st.warning(f"Model file not found. Using fallback logic.")
        return None, False

model, model_loaded = load_model()

# Mapping for categorical features
def prepare_features(data):
    """Prepare features exactly as model expects"""
    
    # Numeric features (6)
    numeric_features = np.array([[
        data['Study_Hours_per_Day'],
        data['Attendance_Rate'],
        data['Assignment_Delay_Days'],
        data['Travel_Time_Minutes'],
        data['Stress_Index'],
        data['GPA']
    ]])
    
    # Categorical features (4)
    internet = 1 if data['Internet_Access'] == 'Yes' else 0
    part_time = 1 if data['Part_Time_Job'] == 'Yes' else 0
    scholarship = 1 if data['Scholarship'] == 'Yes' else 0
    
    semester_map = {'Year 1': 1, 'Year 2': 2, 'Year 3': 3, 'Year 4': 4}
    semester = semester_map.get(data['Semester'], 1)
    
    categorical_features = np.array([[
        internet, part_time, scholarship, semester
    ]])
    
    # Combine all 10 features
    all_features = np.hstack([numeric_features, categorical_features])
    
    return all_features

def predict_dropout(data):
    """Predict using loaded model or fallback"""
    features = prepare_features(data)
    
    if model is not None and model_loaded:
        try:
            prob = model.predict_proba(features)[0][1]
            return prob
        except Exception as e:
            st.warning(f"Model error: {e}")
    
    # Fallback prediction logic based on academic rules
    score = 0
    # GPA (0-4) - most important
    if data['GPA'] < 2.0:
        score += 0.4
    elif data['GPA'] < 2.5:
        score += 0.25
    elif data['GPA'] < 3.0:
        score += 0.15
    
    # Attendance
    if data['Attendance_Rate'] < 60:
        score += 0.3
    elif data['Attendance_Rate'] < 75:
        score += 0.15
    
    # Assignment delay
    if data['Assignment_Delay_Days'] > 7:
        score += 0.2
    elif data['Assignment_Delay_Days'] > 3:
        score += 0.1
    
    # Stress
    if data['Stress_Index'] > 7:
        score += 0.15
    elif data['Stress_Index'] > 5:
        score += 0.05
    
    # Study hours
    if data['Study_Hours_per_Day'] < 2:
        score += 0.15
    
    # Travel time
    if data['Travel_Time_Minutes'] > 90:
        score += 0.1
    
    # Scholarship helps
    if data['Scholarship'] == 'Yes':
        score -= 0.1
    
    # Internet access helps
    if data['Internet_Access'] == 'Yes':
        score -= 0.05
    
    # Part time job slightly increases risk
    if data['Part_Time_Job'] == 'Yes':
        score += 0.05
    
    return min(max(score, 0.01), 0.99)

# Header
st.markdown(f"""
<div class="main-header">
    <h1>EduTrack</h1>
    <div class="subhead">predict · prevent · progress</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "History", "About"])

with tab1:
    col_left, col_right = st.columns([2, 1.2], gap="large")
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Academic Section
        st.markdown('<div class="section-title">Academic Data</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            gpa = st.number_input(
                "GPA",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.1,
                format="%.1f",
                key="gpa"
            )
            
            attendance = st.slider(
                "Attendance Rate (%)",
                min_value=0,
                max_value=100,
                value=85,
                key="attendance"
            )
            
            study_hours = st.slider(
                "Study Hours per Day",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5,
                key="study_hours"
            )
        
        with col2:
            stress = st.slider(
                "Stress Level (1-10)",
                min_value=1,
                max_value=10,
                value=5,
                key="stress"
            )
            
            assignment_delay = st.slider(
                "Assignment Delay (days)",
                min_value=0,
                max_value=14,
                value=2,
                key="delay"
            )
            
            travel_time = st.number_input(
                "Commute Time (minutes)",
                min_value=0,
                max_value=180,
                value=30,
                step=5,
                key="travel"
            )
        
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        # Background Section
        st.markdown('<div class="section-title">Background</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        with col3:
            semester = st.selectbox(
                "Semester",
                ["Year 1", "Year 2", "Year 3", "Year 4"],
                key="semester"
            )
            
            part_time = st.selectbox(
                "Part-time Job",
                ["No", "Yes"],
                key="parttime"
            )
        
        with col4:
            internet = st.selectbox(
                "Internet Access",
                ["Yes", "No"],
                key="internet"
            )
            
            scholarship = st.selectbox(
                "Scholarship",
                ["No", "Yes"],
                key="scholarship"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown('<div class="result-label">PREDICTION</div>', unsafe_allow_html=True)
        
        predict_btn = st.button("Analyze", use_container_width=True, key="predict_btn")
        
        if predict_btn:
            input_data = {
                'GPA': gpa,
                'Stress_Index': stress,
                'Assignment_Delay_Days': assignment_delay,
                'Attendance_Rate': attendance,
                'Study_Hours_per_Day': study_hours,
                'Travel_Time_Minutes': travel_time,
                'Part_Time_Job': part_time,
                'Internet_Access': internet,
                'Scholarship': scholarship,
                'Semester': semester
            }
            
            prob = predict_dropout(input_data)
            
            if prob >= 0.6:
                risk = "High"
                risk_class = "risk-high"
                recommendation = "Immediate academic advising recommended. Student shows multiple risk factors including low GPA and poor attendance."
            elif prob >= 0.35:
                risk = "Moderate"
                risk_class = "risk-medium"
                recommendation = "Monitor progress. Consider connecting student with tutoring services and academic support."
            else:
                risk = "Low"
                risk_class = "risk-low"
                recommendation = "Student appears on track. Continue current support and regular check-ins."
            
            st.markdown(f'<div class="result-value {risk_class}">{prob:.0%}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-label">{risk} RISK</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="recommendation">{recommendation}</div>', unsafe_allow_html=True)
            
            # Save to history
            st.session_state.history.append({
                'GPA': gpa,
                'Attendance': attendance,
                'Study_Hours': study_hours,
                'Delay': assignment_delay,
                'Stress': stress,
                'Probability': prob,
                'Risk': risk,
                'Semester': semester
            })
            
            if len(st.session_state.history) > 20:
                st.session_state.history = st.session_state.history[-20:]
        
        else:
            st.markdown("""
                <div class="recommendation" style="margin-top: 1rem; text-align: center;">
                    Fill in student data<br>and click "Analyze"
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # Stats row
        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        
        with col_s1:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{len(df)}</div>
                    <div class="stat-label">Total Predictions</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_s2:
            high_risk = len(df[df['Risk'] == 'High'])
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number" style="color: #c0392b;">{high_risk}</div>
                    <div class="stat-label">High Risk</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_s3:
            avg_prob = df['Probability'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_prob:.0%}</div>
                    <div class="stat-label">Avg Probability</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_s4:
            avg_gpa = df['GPA'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-number">{avg_gpa:.2f}</div>
                    <div class="stat-label">Avg GPA</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        
        colors = ['#c0392b' if p >= 0.6 else '#e67e22' if p >= 0.35 else COLORS['dark_moss'] 
                  for p in df['Probability']]
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(df) + 1)),
            y=df['Probability'],
            marker_color=colors,
            text=[f'{p:.0%}' for p in df['Probability']],
            textposition='outside',
            name='Risk Probability'
        ))
        
        fig.update_layout(
            title="Risk Trend",
            xaxis_title="Prediction Sequence",
            yaxis_title="Dropout Probability",
            yaxis_tickformat='.0%',
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="Inter, sans-serif", size=11, color=COLORS['licorice'])
        )
        
        fig.update_xaxes(gridcolor=COLORS['dutch_white'], showgrid=True)
        fig.update_yaxes(gridcolor=COLORS['dutch_white'], showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.markdown("### Recent Predictions")
        
        display_df = df[['Semester', 'GPA', 'Attendance', 'Study_Hours', 'Delay', 'Stress', 'Risk']].copy()
        display_df.columns = ['Semester', 'GPA', 'Attend %', 'Study Hrs', 'Delay', 'Stress', 'Risk']
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("Clear History", use_container_width=True, key="clear_btn"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions yet. Go to the Predict tab to get started.")

with tab3:
    st.markdown("""
    <div class="card">
        <div class="section-title">Model Information</div>
        
        <p style="font-family: 'Inter', sans-serif; font-size: 0.85rem; line-height: 1.5; color: #1C150F;">
        This predictor uses <strong>Logistic Regression</strong> trained on 10 features to identify
        students at risk of dropping out.
        </p>
        
        <div class="custom-divider"></div>
        
        <div class="section-title">Features Used</div>
        
        <table style="width: 100%; font-family: 'Inter', sans-serif; font-size: 0.8rem; border-collapse: collapse;">
            <tr style="border-bottom: 1px solid #D7CEAF;">
                <th style="text-align: left; padding: 0.5rem 0;">Numeric</th>
                <th style="text-align: left; padding: 0.5rem 0;">Categorical</th>
            </tr>
            <tr style="border-bottom: 1px solid #D7CEAF;">
                <td style="padding: 0.5rem 0;">• GPA</td>
                <td style="padding: 0.5rem 0;">• Internet Access</td>
            </tr>
            <tr style="border-bottom: 1px solid #D7CEAF;">
                <td style="padding: 0.5rem 0;">• Stress Index</td>
                <td style="padding: 0.5rem 0;">• Part-time Job</td>
            </tr>
            <tr style="border-bottom: 1px solid #D7CEAF;">
                <td style="padding: 0.5rem 0;">• Assignment Delay</td>
                <td style="padding: 0.5rem 0;">• Scholarship</td>
            </tr>
            <tr style="border-bottom: 1px solid #D7CEAF;">
                <td style="padding: 0.5rem 0;">• Attendance Rate</td>
                <td style="padding: 0.5rem 0;">• Semester</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;">• Study Hours</td>
                <td></td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;">• Travel Time</td>
                <td></td>
            </tr>
        </table>
        
        <div class="custom-divider"></div>
        
        <div class="section-title">Risk Levels</div>
        
        <ul style="font-family: 'Inter', sans-serif; font-size: 0.8rem; line-height: 1.8; color: #1C150F;">
            <li><span style="color: #c0392b; font-weight: 600;">High (≥60%)</span> — Immediate intervention needed</li>
            <li><span style="color: #e67e22; font-weight: 600;">Moderate (35-59%)</span> — Monitor and provide support</li>
            <li><span style="color: #5C6844; font-weight: 600;">Low (&lt;35%)</span> — Student on track</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown(f"""
<div class="footer">
    EduTrack — Dropout Prediction System<br>
    Built with Logistic Regression | 10 Features
</div>
""", unsafe_allow_html=True)
