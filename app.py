
import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>
.stApp {
    background: #0b1120;
    color: #f1f5f9;
}

[data-testid="stSidebar"] {
    background: #111c31;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
}

.subtitle {
    color: #9caec5;
    font-size: 16px;
}

.hero {
    background: linear-gradient(120deg, #142c50, #174a82);
    padding: 30px;
    border-radius: 18px;
    margin-bottom: 25px;
    border: 1px solid #28558a;
}

.section-title {
    font-size: 23px;
    font-weight: 700;
    color: #ffffff;
    margin-top: 20px;
}

.info-card {
    background: #142238;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #263c59;
    min-height: 110px;
}

.result-card {
    background: linear-gradient(135deg, #123b5c, #145b78);
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 1px solid #277b9d;
    margin-top: 20px;
}

.result-price {
    font-size: 40px;
    font-weight: 800;
    color: #65e6c4;
}

.stButton > button {
    background: #1685e5;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 25px;
    font-weight: bold;
    width: 100%;
}

.stButton > button:hover {
    background: #0868bd;
    color: white;
}

div[data-testid="stMetric"] {
    background: #142238;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263c59;
}
</style>
""", unsafe_allow_html=True)


# ---------------- LOAD FILES ----------------

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "best_model.pkl")


@st.cache_data
def load_data():
    with open(BASE_DIR / "model_features.json", "r") as file:
        features = json.load(file)

    df = pd.read_csv(BASE_DIR / "laptops_cleaned.csv")
    return df, features


try:
    model = load_model()
    df, features = load_data()

    with open(BASE_DIR / "best_model_name.txt", "r") as file:
        model_name = file.read().strip()

except Exception as e:
    st.error("Unable to load the project files.")
    st.write("Check that all model and dataset files are in the same folder as app.py.")
    st.code(str(e))
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 💻 LaptopPredict")
    st.caption("Machine Learning Price Estimator")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["🏠 Home", "💰 Price Prediction", "📊 Dataset Overview", "ℹ️ About Project"]
    )

    st.divider()

    st.markdown("### 🤖 Model Information")
    st.success(model_name)

    st.caption("Powered by Python, Scikit-learn and Streamlit")


# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.markdown("""
    <div class="hero">
        <div class="main-title">Find the Right Laptop Price</div>
        <p class="subtitle">
        Smart laptop price estimation using Machine Learning.
        Enter laptop specifications and get an estimated price instantly.
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
        <h2>⚡</h2>
        <h3>Fast Prediction</h3>
        <p>Get an estimated laptop price in seconds.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
        <h2>🎯</h2>
        <h3>ML Powered</h3>
        <p>Predictions using your trained regression model.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
        <h2>💻</h2>
        <h3>Custom Specifications</h3>
        <p>Choose specifications to estimate laptop prices.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("### Get Started")
    st.write("Go to Price Prediction from the sidebar to estimate your laptop's price.")

    if st.button("Start Price Prediction"):
        st.switch_page("app.py")


# ---------------- PREDICTION PAGE ----------------

elif page == "💰 Price Prediction":

    st.markdown('<div class="main-title">Laptop Price Prediction</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<p class="subtitle">Enter the laptop specifications below.</p>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="section-title">⚙️ Laptop Specifications</div>',
                unsafe_allow_html=True)

    input_data = {}

    with st.form("prediction_form"):

        numeric_features = [
            col for col in features
            if pd.api.types.is_numeric_dtype(df[col])
        ]

        categorical_features = [
            col for col in features
            if col not in numeric_features
        ]

        st.markdown("#### 🔢 Numeric Specifications")

        numeric_columns = st.columns(2)

        for i, col in enumerate(numeric_features):
            with numeric_columns[i % 2]:
                median_value = df[col].median()

                if pd.isna(median_value):
                    median_value = 0

                input_data[col] = st.number_input(
                    col.replace("_", " ").title(),
                    value=float(median_value)
                )

        st.markdown("---")
        st.markdown("#### 🏷️ Laptop Details")

        categorical_columns = st.columns(2)

        for i, col in enumerate(categorical_features):
            with categorical_columns[i % 2]:
                options = (
                    df[col]
                    .dropna()
                    .astype(str)
                    .unique()
                    .tolist()
                )

                if options:
                    input_data[col] = st.selectbox(
                        col.replace("_", " ").title(),
                        options
                    )
                else:
                    input_data[col] = st.text_input(
                        col.replace("_", " ").title(),
                        value="Unknown"
                    )

        st.markdown("<br>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "🚀 Predict Laptop Price",
            use_container_width=True
        )

    if submitted:
        try:
            input_df = pd.DataFrame([input_data], columns=features)
            prediction = float(model.predict(input_df)[0])

            st.markdown("""
            <div class="result-card">
                <h3>✨ Estimated Laptop Price</h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(
                f'<div class="result-card"><div class="result-price">₹ {prediction:,.2f}</div><p>Predicted Price</p></div>',
                unsafe_allow_html=True
            )

            st.success("Price prediction completed successfully!")

            with st.expander("View Entered Specifications"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed. Please check the input specifications.")
            st.code(str(e))


# ---------------- DATASET PAGE ----------------

elif page == "📊 Dataset Overview":

    st.markdown('<div class="main-title">Dataset Overview</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<p class="subtitle">Explore the laptop dataset used in this project.</p>',
        unsafe_allow_html=True
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(features))
    c3.metric("Missing Values", int(df[features].isnull().sum().sum()))

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.markdown("### Dataset Statistics")
    st.dataframe(df[features].describe(), use_container_width=True)

    if "price" in df.columns:
        st.markdown("### Price Distribution")
        st.bar_chart(df["price"].value_counts().sort_index())


# ---------------- ABOUT PAGE ----------------

elif page == "ℹ️ About Project":

    st.markdown('<div class="main-title">About This Project</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <h2>💻 Laptop Price Prediction Using Machine Learning</h2>
        <p class="subtitle">
        This project predicts laptop prices based on their technical
        specifications using regression algorithms.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.markdown("### Machine Learning Models")

    st.write("""
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Decision Tree Regression
    - Random Forest Regression
    """)

    st.markdown("### Best Performing Model")
    st.success(model_name)

    st.caption("Academic Machine Learning Project")
