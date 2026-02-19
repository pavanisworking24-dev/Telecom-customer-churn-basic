import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-top: 0;
    }
    .risk-badge-low {
        background-color: #28a745;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-badge-medium {
        background-color: #ffc107;
        color: black;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .risk-badge-high {
        background-color: #dc3545;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
    }
    .metric-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# LOAD SAVED FILES
# -------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    threshold = joblib.load("threshold.pkl")
    all_features = joblib.load("all_features.pkl")
    selected_features = joblib.load("selected_features.pkl")
    return model, scaler, threshold, all_features, selected_features

model, scaler, THRESHOLD, all_features, selected_features = load_artifacts()

# -------------------------------
# SIDEBAR - INPUT FORM
# -------------------------------
st.sidebar.image("https://img.icons8.com/color/96/000000/customer-insight.png", width=100)
st.sidebar.title("📋 Customer Details")
st.sidebar.markdown("Fill in the information below to predict churn risk.")

with st.sidebar.form("input_form"):
    st.markdown("### 👤 Personal")
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
    with col2:
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
    
    st.markdown("### 📞 Account & Services")
    col1, col2 = st.columns(2)
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    with col2:
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        MonthlyCharges = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges ($)", 0.0, 10000.0, 2000.0)
    
    st.markdown("### 🔒 Online Services")
    col1, col2 = st.columns(2)
    with col1:
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    with col2:
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.markdown("### 📄 Contract & Billing")
    col1, col2 = st.columns(2)
    with col1:
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
    with col2:
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    
    submitted = st.form_submit_button("🚀 Predict Churn Risk", use_container_width=True)

# -------------------------------
# MAIN AREA
# -------------------------------
st.markdown("<h1 class='main-header'>📊 Customer Churn Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Logistic Regression with L1 feature selection & optimized threshold</p>", unsafe_allow_html=True)

if submitted:
    # --- Build input dataframe ---
    input_dict = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    input_df = pd.DataFrame([input_dict])

    # --- Preprocessing (same as training) ---
    # One-hot encode
    input_df = pd.get_dummies(input_df)
    # Align with all_features (fill missing with 0)
    input_df = input_df.reindex(columns=all_features, fill_value=0)
    # Scale
    input_scaled = scaler.transform(input_df)
    # Select L1-selected features
    mask = np.isin(all_features, selected_features)
    input_selected = input_scaled[:, mask]

    # --- Predict ---
    probability = model.predict_proba(input_selected)[0][1]
    prediction = 1 if probability >= THRESHOLD else 0

    # --- Risk level ---
    if probability < 0.3:
        risk_level = "Low"
        risk_badge_class = "risk-badge-low"
    elif probability < 0.6:
        risk_level = "Medium"
        risk_badge_class = "risk-badge-medium"
    else:
        risk_level = "High"
        risk_badge_class = "risk-badge-high"

    # --- Display results in tabs ---
    tab1, tab2, tab3 = st.tabs(["📈 Prediction", "🔍 Feature Impact", "ℹ️ Model Info"])

    with tab1:
        # Top row: metrics using st.metric with custom background
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Churn Probability", f"{probability:.2%}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Threshold", f"{THRESHOLD:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            verdict = "⚠️ High Risk" if prediction == 1 else "✅ Low Risk"
            st.metric("Prediction", verdict)
            st.markdown('</div>', unsafe_allow_html=True)

        # Gauge chart with plotly
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={'suffix': "%", 'font': {'size': 40, 'color': 'black'}},
            title={'text': "Churn Risk", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': "darkred" if prediction == 1 else "green"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 60], 'color': '#fff3cd'},
                    {'range': [60, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': THRESHOLD * 100
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Risk badge
        st.markdown(f"**Risk Level:** <span class='{risk_badge_class}'>{risk_level}</span>", unsafe_allow_html=True)
        st.caption(f"*Threshold used: {THRESHOLD:.2f} (optimized for recall)*")

    with tab2:
        # Compute feature contributions (coefficient * scaled feature value)
        coef = model.coef_[0]
        contributions = coef * input_selected[0]
        contrib_df = pd.DataFrame({
            'Feature': selected_features,
            'Contribution': contributions
        }).sort_values('Contribution', key=abs, ascending=False).head(10)

        # Bar chart
        fig2 = px.bar(
            contrib_df,
            x='Contribution',
            y='Feature',
            orientation='h',
            title='Top 10 Feature Contributions',
            labels={'Contribution': 'Impact on Churn Probability (log-odds)'},
            color='Contribution',
            color_continuous_scale=['green', 'lightgray', 'red'],
            range_color=[-contrib_df['Contribution'].abs().max(), contrib_df['Contribution'].abs().max()]
        )
        fig2.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("""
        **How to read:**  
        - Positive contribution (red) pushes the prediction toward **churn**.  
        - Negative contribution (green) pushes toward **stay**.  
        - Values are in log-odds scale (higher absolute = stronger influence).
        """)

    with tab3:
        st.markdown("""
        ### Model Overview
        - **Algorithm:** Logistic Regression with L1 regularization (feature selection)
        - **Training data:** Telecom customer churn dataset
        - **Feature selection:** L1 penalty reduced original features to **{}** selected features.
        - **Threshold tuning:** Optimized for recall (set to **{:.2f}**) to catch as many churners as possible.
        - **Performance metrics (on test set):**  
          - Recall: **{}**  
          - Precision: **{}**  
          - F1-Score: **{}**
        """.format(len(selected_features), THRESHOLD, "0.82", "0.61", "0.70"))  # Replace with actual numbers

        if st.checkbox("Show selected features"):
            st.write(selected_features)

else:
    st.info("👈 Fill in the customer details in the sidebar and click **Predict Churn Risk** to see the result.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | Logistic Regression model")