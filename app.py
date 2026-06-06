# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

# Page config
st.set_page_config(
    page_title="Dropout Predictor",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS - minimal & professional
st.markdown("""
<style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        background: #f5f5f5;
    }
    
    .main {
        max-width: 1400px;
        margin: 0 auto;
        padding: 2rem;
    }
    
    h1 {
        font-size: 2rem;
        font-weight: 600;
        color: #1a1a1a;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    
    .subhead {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e0e0e0;
        padding-bottom: 1rem;
    }
    
    .card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #eaeaea;
    }
    
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        border: 1px solid #eaeaea;
    }
    
    .result-number {
        font-size: 3.5rem;
        font-weight: 600;
        margin: 1rem 0;
    }
    
    .result-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
    }
    
    .risk-high {
        color: #dc2626;
    }
    
    .risk-medium {
        color: #f59e0b;
    }
    
    .risk-low {
        color: #10b981;
    }
    
    .recommendation {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        font-size: 0.9rem;
        color: #444;
    }
    
    .stat-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #eaeaea;
    }
    
    .stat-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
    }
    
    .stat-label {
        font-size: 0.75rem;
        color: #888;
        margin-top: 0.25rem;
    }
    
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 1px solid #eaeaea;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #999;
        font-size: 0.75rem;
        border-top: 1px solid #eaeaea;
        margin-top: 2rem;
    }
    
    /* Form styling */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 8px;
        border: 1px solid #ddd;
        padding: 0.5rem 0.75rem;
    }
    
    .stButton > button {
        background: #1a1a1a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        width: 100%;
        transition: all 0.2s;
    }
    
    .stButton > button:hover {
        background: #333;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid #eaeaea;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-size: 0.9rem;
        color: #666;
        padding: 0.5rem 0;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1a1a1a;
        border-bottom: 2px solid #1a1a1a;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #e0e0e0;
    }
    
    .stSlider > div > div > div > div {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []

# Load or train model
@st.cache_resource
def get_model():
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except:
        # Train a simple model if pickle not found
        from sklearn.linear_model import LogisticRegression
        
        # Generate synthetic training data based on logical patterns
        np.random.seed(42)
        n_samples = 1000
        
        X_train = np.random.randn(n_samples, 9)
        # GPA (index 0): lower GPA = higher dropout risk
        # Stress (1): higher stress = higher risk
        # Delay (2): more delay = higher risk
        # Attendance (3): lower attendance = higher risk
        # Study hours (4): fewer hours = higher risk
        # Travel (5): longer travel = higher risk
        # Part time job (6): yes = slightly higher risk
        # Internet access (7): no = higher risk
        # Scholarship (8): no = higher risk
        
        log_odds = (
            -1.5 * X_train[:, 0] +  # GPA
            0.3 * X_train[:, 1] +   # Stress
            0.4 * X_train[:, 2] +   # Delay
            -0.3 * X_train[:, 3] +  # Attendance
            -0.2 * X_train[:, 4] +  # Study hours
            0.15 * X_train[:, 5]    # Travel
        )
        
        prob = 1 / (1 + np.exp(-log_odds))
        y_train = (prob > 0.5).astype(int)
        
        model = LogisticRegression(C=1.0, max_iter=1000)
        model.fit(X_train, y_train)
        return model

model = get_model()

def prepare_features(data):
    """Prepare features for prediction"""
    features = np.array([[
        data['gpa'],
        data['stress'],
        data['assignment_delay'],
        data['attendance'] / 100,  # normalize to 0-1
        data['study_hours'] / 12,   # normalize
        data['travel_time'] / 120,  # normalize
        1 if data['part_time'] == 'Yes' else 0,
        1 if data['internet'] == 'Yes' else 0,
        1 if data['scholarship'] == 'Yes' else 0,
    ]])
    return features

def predict(data):
    """Make prediction"""
    features = prepare_features(data)
    prob = model.predict_proba(features)[0][1]
    return prob

# Main layout
st.markdown('<div class="main">', unsafe_allow_html=True)

# Header
st.markdown("<h1>Student Dropout Predictor</h1>", unsafe_allow_html=True)
st.markdown('<div class="subhead">Early identification system for at-risk students</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Predict", "History", "Guide"])

with tab1:
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Academic Information
        st.markdown("### Academic Data")
        col1, col2 = st.columns(2)
        
        with col1:
            gpa = st.number_input(
                "GPA (0.0 - 4.0)",
                min_value=0.0,
                max_value=4.0,
                value=3.0,
                step=0.1,
                format="%.1f"
            )
            
            attendance = st.slider(
                "Attendance Rate (%)",
                min_value=0,
                max_value=100,
                value=85
            )
            
            study_hours = st.slider(
                "Study Hours per Day",
                min_value=0.0,
                max_value=12.0,
                value=4.0,
                step=0.5
            )
        
        with col2:
            stress = st.slider(
                "Stress Level (1-10)",
                min_value=1,
                max_value=10,
                value=5
            )
            
            assignment_delay = st.slider(
                "Assignment Delay (days)",
                min_value=0,
                max_value=14,
                value=2
            )
            
            travel_time = st.number_input(
                "Commute Time (minutes)",
                min_value=0,
                max_value=180,
                value=30,
                step=5
            )
        
        # Socioeconomic Information
        st.markdown("### Background Information")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            semester = st.selectbox(
                "Current Semester",
                ["Year 1", "Year 2", "Year 3", "Year 4"]
            )
        
        with col4:
            part_time = st.selectbox(
                "Part-time Job",
                ["No", "Yes"]
            )
        
        with col5:
            scholarship = st.selectbox(
                "Scholarship",
                ["No", "Yes"]
            )
        
        internet = st.selectbox(
            "Internet Access",
            ["Yes", "No"],
            index=0
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### Prediction Result")
        
        predict_btn = st.button("Analyze Risk", use_container_width=True)
        
        if predict_btn:
            input_data = {
                'gpa': gpa,
                'stress': stress,
                'assignment_delay': assignment_delay,
                'attendance': attendance,
                'study_hours': study_hours,
                'travel_time': travel_time,
                'part_time': part_time,
                'internet': internet,
                'scholarship': scholarship,
                'semester': semester
            }
            
            prob = predict(input_data)
            
            if prob >= 0.6:
                risk = "High"
                risk_class = "risk-high"
                recommendation = "Immediate academic advising recommended. Student shows multiple risk factors."
            elif prob >= 0.35:
                risk = "Moderate"
                risk_class = "risk-medium"
                recommendation = "Monitor progress. Consider academic support services."
            else:
                risk = "Low"
                risk_class = "risk-low"
                recommendation = "Student appears on track. Continue current support."
            
            st.markdown(f"""
                <div class="result-label">DROPOUT RISK</div>
                <div class="result-number {risk_class}">{prob:.0%}</div>
                <div class="result-label" style="margin-top: 0.5rem;">{risk} RISK</div>
            """, unsafe_allow_html=True)
            
            st.markdown(f'<div class="recommendation">{recommendation}</div>', unsafe_allow_html=True)
            
            # Save to history
            st.session_state.history.append({
                'gpa': gpa,
                'attendance': attendance,
                'study_hours': study_hours,
                'probability': prob,
                'risk': risk,
                'semester': semester
            })
            
            # Keep only last 20
            if len(st.session_state.history) > 20:
                st.session_state.history = st.session_state.history[-20:]
        
        else:
            st.markdown("""
                <div class="recommendation" style="margin-top: 1rem;">
                    Fill in student data and click "Analyze Risk" to get prediction.
                </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    if len(st.session_state.history) > 0:
        df = pd.DataFrame(st.session_state.history)
        
        # Stats
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{len(df)}</div>
                    <div class="stat-label">Total Predictions</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_b:
            high_risk = len(df[df['risk'] == 'High'])
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value" style="color: #dc2626;">{high_risk}</div>
                    <div class="stat-label">High Risk Cases</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_c:
            avg_prob = df['probability'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg_prob:.0%}</div>
                    <div class="stat-label">Avg. Probability</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col_d:
            avg_gpa = df['gpa'].mean()
            st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-value">{avg_gpa:.2f}</div>
                    <div class="stat-label">Avg. GPA</div>
                </div>
            """, unsafe_allow_html=True)
        
        # Chart
        fig = go.Figure()
        
        colors = ['#dc2626' if p >= 0.6 else '#f59e0b' if p >= 0.35 else '#10b981' 
                  for p in df['probability']]
        
        fig.add_trace(go.Bar(
            x=list(range(1, len(df) + 1)),
            y=df['probability'],
            marker_color=colors,
            text=[f'{p:.0%}' for p in df['probability']],
            textposition='outside',
            name='Risk Probability'
        ))
        
        fig.update_layout(
            title="Risk Trend Over Time",
            xaxis_title="Prediction Sequence",
            yaxis_title="Dropout Probability",
            yaxis_tickformat='.0%',
            height=350,
            margin=dict(l=40, r=40, t=60, b=40),
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(family="system-ui", size=12, color="#333")
        )
        
        fig.update_xaxes(gridcolor='#f0f0f0', showgrid=True)
        fig.update_yaxes(gridcolor='#f0f0f0', showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # History table
        st.markdown("### Prediction History")
        
        display_df = df[['semester', 'gpa', 'attendance', 'study_hours', 'risk']].copy()
        display_df.columns = ['Semester', 'GPA', 'Attendance', 'Study Hrs', 'Risk']
        display_df['Attendance'] = display_df['Attendance'].astype(str) + '%'
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()
    
    else:
        st.info("No predictions yet. Go to the Predict tab to get started.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    
    st.markdown("""
    ### About the Model
    
    This predictor uses **Logistic Regression** trained on student academic and demographic data to identify dropout risk factors.
    
    ### Key Risk Factors
    
    | Factor | Impact |
    |--------|--------|
    | Low GPA | High |
    | Poor Attendance | High |
    | Assignment Delays | High |
    | High Stress | Medium |
    | Low Study Hours | Medium |
    | Long Commute | Low |
    
    ### Interpretation
    
    - **High Risk (≥60%)**: Immediate intervention recommended
    - **Moderate Risk (35-59%)**: Monitor and provide support
    - **Low Risk (<35%)**: Student appears on track
    
    ### Recommendations by Risk Level
    
    **High Risk**
    - Schedule academic advising session
    - Connect with student support services
    - Consider reduced course load
    - Weekly progress check-ins
    
    **Moderate Risk**
    - Encourage tutoring services
    - Monitor attendance pattern
    - Check in every 2-3 weeks
    
    **Low Risk**
    - Maintain current support
    - Recognize good standing
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class="footer">
        Dropout Prediction System | Logistic Regression Model
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
