
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

.hero {
    background: linear-gradient(120deg, #142c50, #174a82);
    padding: 35px;
    border-radius: 18px;
    margin-bottom: 28px;
    border: 1px solid #28558a;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    color: white;
}

.subtitle {
    color: #c1d4ec;
    font-size: 16px;
    margin-top: 12px;
}

.info-card {
    background: #142238;
    padding: 24px;
    border-radius: 15px;
    border: 1px solid #263c59;
    min-height: 220px;
}

.info-card h2 {
    font-size: 35px;
}

.info-card h3 {
    color: white;
    font-size: 23px;
}

.info-card p {
    color: #e2e8f0;
    font-size: 16px;
    line-height: 1.6;
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

.stButton > button,
.stFormSubmitButton > button {
    background: #1685e5;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: bold;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
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


# ---------------- LOAD MODEL AND DATA ----------------

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
    st.write("Check that all required files are present beside app.py.")
    st.code(str(e))
    st.stop()


# ---------------- SIDEBAR ----------------

with st.sidebar:

    st.markdown("## 💻 LaptopPredict")
    st.caption("Machine Learning Price Estimator")

    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "🏠 Home",
            "💰 Price Prediction",
            "📊 Dataset Overview",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.markdown("### 🤖 Model Information")
    st.success(model_name)

    st.caption("Powered by Python, Scikit-learn and Streamlit")


# ---------------- HOME PAGE ----------------

if page == "🏠 Home":

    st.html("""
    <div class="hero">
        <div class="main-title">
            Find the Right Laptop Price
        </div>

        <p class="subtitle">
            Smart laptop price estimation using Machine Learning.
            Enter laptop specifications and get an estimated price instantly.
        </p>
    </div>
    """)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.html("""
        <div class="info-card">
            <h2>⚡</h2>
            <h3>Fast Prediction</h3>
            <p>
                Get an estimated laptop price in seconds.
            </p>
        </div>
        """)

    with c2:
        st.html("""
        <div class="info-card">
            <h2>🎯</h2>
            <h3>ML Powered</h3>
            <p>
                Predictions using your trained regression model.
            </p>
        </div>
        """)

    with c3:
        st.html("""
        <div class="info-card">
            <h2>💻</h2>
            <h3>Custom Specifications</h3>
            <p>
                Choose specifications to estimate laptop prices.
            </p>
        </div>
        """)


# ---------------- PRICE PREDICTION PAGE ----------------

elif page == "💰 Price Prediction":

    st.title("💰 Laptop Price Prediction")

    st.write("Enter the laptop specifications to estimate its price.")

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

        st.subheader("Numeric Specifications")

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

        st.divider()

        st.subheader("Laptop Details")

        categorical_columns = st.columns(2)

        for i, col in enumerate(categorical_features):

            with categorical_columns[i % 2]:

                options = (
                    df[col]
                    .dropna()
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
                        col.replace("_", " ").title()
                    )

        submitted = st.form_submit_button(
            "🚀 Predict Laptop Price",
            use_container_width=True
        )

    if submitted:

        try:

            input_df = pd.DataFrame(
                [input_data],
                columns=features
            )

            prediction = float(model.predict(input_df)[0])

            st.html(f"""
            <div class="result-card">
                <h3>✨ Estimated Laptop Price</h3>
                <div class="result-price">
                    ₹ {prediction:,.2f}
                </div>
                <p>Predicted Laptop Price</p>
            </div>
            """)

            st.success("Price prediction completed successfully!")

            with st.expander("View Entered Specifications"):
                st.dataframe(input_df, use_container_width=True)

        except Exception as e:

            st.error("Prediction failed. Please check the specifications.")
            st.code(str(e))


# ---------------- DATASET OVERVIEW PAGE ----------------

elif page == "📊 Dataset Overview":

    st.title("📊 Dataset Overview")

    st.write("Explore the dataset used for laptop price prediction.")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Total Features", len(features))
    c3.metric(
        "Missing Values",
        int(df[features].isnull().sum().sum())
    )

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(20),
        use_container_width=True
    )

    st.subheader("Dataset Statistics")

    st.dataframe(
        df[features].describe(),
        use_container_width=True
    )

    if "price" in df.columns:

        st.subheader("Price Distribution")

        st.bar_chart(df["price"])


# ---------------- ABOUT PROJECT PAGE ----------------

elif page == "ℹ️ About Project":

    st.title("ℹ️ About This Project")

    st.html("""
    <div class="hero">
        <h2>💻 Laptop Price Prediction Using Machine Learning</h2>
        <p class="subtitle">
            This project predicts laptop prices based on their
            technical specifications using regression algorithms.
        </p>
    </div>
    """)

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.subheader("Machine Learning Algorithms")

    st.write("""
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - Decision Tree Regression
    - Random Forest Regression
    """)

    st.subheader("Best Performing Model")

    st.success(model_name)

    st.caption("Academic Machine Learning Project")
