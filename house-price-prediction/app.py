
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
from pathlib import Path

# Find the saved machine learning model
project_folder = Path(__file__).resolve().parent
model_path = project_folder / "model" / "random_forest_model.joblib"

# Set up the webpage
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# Simple white and green design

# Simple white and green design
st.markdown("""
<style>
/* Main page */
.stApp {
    background-color: #FFFFFF;
    color: #222222;
}

/* Headings */
h1, h2, h3, h4 {
    color: #1F2937 !important;
}

/* Normal text and input labels */
.stApp p,
.stApp label,
.stApp [data-testid="stMarkdownContainer"] p {
    color: #374151;
}

/* Sidebar background and text */
[data-testid="stSidebar"] {
    background-color: #F3F7F4;
}

[data-testid="stSidebar"] * {
    color: #26352A;
}

/* Input fields */
[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #222222 !important;
    border: 1px solid #D1D5DB;
    border-radius: 8px;
}

/* Input buttons */
[data-testid="stNumberInput"] button {
    color: #222222 !important;
}

/* Form */
[data-testid="stForm"] {
    background-color: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 20px;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #F8FAFC;
    border: 1px solid #E5E7EB;
    border-radius: 10px;
    padding: 15px;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #4B5563 !important;
}

[data-testid="stMetricValue"] {
    color: #4D7C0F !important;
}

/* Green buttons */
.stButton button,
.stFormSubmitButton button {
    background-color: #65A30D !important;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    font-weight: bold;
}

.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #4D7C0F !important;
    color: #FFFFFF !important;
}

/* Data tables */
[data-testid="stDataFrame"] {
    border: 1px solid #E5E7EB;
    border-radius: 8px;
}

/* Information boxes */
[data-testid="stAlert"] {
    border-radius: 8px;
}

/* Footer and captions */
.stCaption,
[data-testid="stCaptionContainer"] {
    color: #6B7280 !important;
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load(model_path)


try:
    model = load_model()
except Exception:
    st.error("Model not found. Please check the model folder.")
    st.stop()


# Names of the features used during training
feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude"
]


# Sidebar menu
st.sidebar.title("🏡 House Predictor")
st.sidebar.write("A beginner-friendly Machine Learning project.")

page = st.sidebar.radio(
    "Select a page",
    ["Predict Price", "Model Performance", "Feature Importance"]
)

st.sidebar.divider()
st.sidebar.write("**Model:** Random Forest")
st.sidebar.write("**Language:** Python")
st.sidebar.write("**App:** Streamlit")


# --------------------------------
# PAGE 1: PREDICT HOUSE PRICE
# --------------------------------
if page == "Predict Price":

    st.title("🏡 House Price Predictor")

    st.write(
        "Enter the details below to estimate the median "
        "house value for an area in California."
    )

    st.info(
        "This predicts an area's median housing value, "
        "not the exact selling price of an individual house."
    )

    with st.form("house_form"):

        st.subheader("Enter House Details")

        col1, col2 = st.columns(2)

        with col1:
            median_income = st.number_input(
                "Median Income (in $10,000 units)",
                min_value=0.0,
                value=4.0,
                step=0.1
            )

            house_age = st.number_input(
                "Median House Age (years)",
                min_value=0.0,
                value=25.0,
                step=1.0
            )

            average_rooms = st.number_input(
                "Average Rooms per Household",
                min_value=0.1,
                value=5.0,
                step=0.1
            )

            average_bedrooms = st.number_input(
                "Average Bedrooms per Household",
                min_value=0.1,
                value=1.0,
                step=0.1
            )

        with col2:
            population = st.number_input(
                "Area Population",
                min_value=1.0,
                value=1000.0,
                step=100.0
            )

            average_occupants = st.number_input(
                "Average Occupants per Household",
                min_value=0.1,
                value=3.0,
                step=0.1
            )

            latitude = st.number_input(
                "Latitude",
                min_value=32.0,
                max_value=42.0,
                value=34.05,
                step=0.01,
                format="%.2f"
            )

            longitude = st.number_input(
                "Longitude",
                min_value=-125.0,
                max_value=-114.0,
                value=-118.25,
                step=0.01,
                format="%.2f"
            )

        predict_button = st.form_submit_button(
            "Predict House Price",
            use_container_width=True
        )

    if predict_button:

        # Put the inputs into a table for the model
        input_data = pd.DataFrame(
            [[
                median_income,
                house_age,
                average_rooms,
                average_bedrooms,
                population,
                average_occupants,
                latitude,
                longitude
            ]],
            columns=feature_names
        )

        # Ask the model to predict the value
        try:
            prediction = model.predict(input_data)[0]

            # The target uses units of $100,000
            estimated_price = prediction * 100000

            st.success("Prediction completed!")

            st.subheader("Estimated Median House Value")

            st.markdown(
                f"""
                <div style="
                    background-color: #F0FDF4;
                    border: 1px solid #BBF7D0;
                    border-radius: 12px;
                    padding: 25px;
                    text-align: center;
                ">
                    <h1 style="
                        color: #3F6212;
                        font-size: 36px;
                    ">
                        ${estimated_price:,.0f}
                    </h1>
                    <p>Estimated value for the selected area</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.caption(
                "The dataset caps its target at approximately "
                "$500,000. Actual property values may differ."
            )

            with st.expander("View the details you entered"):
                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------
# PAGE 2: MODEL PERFORMANCE
# --------------------------------
elif page == "Model Performance":

    st.title("Model Performance")

    st.write(
        "These are the results from the test set used "
        "in the original project evaluation."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Linear Regression R²", "57.58%")
        st.metric("Linear Regression MAE", "$53,320")
        st.metric("Linear Regression RMSE", "$74,560")

    with col2:
        st.metric("Random Forest R²", "80.51%")
        st.metric("Random Forest MAE", "$32,750")
        st.metric("Random Forest RMSE", "$50,530")

    st.caption(
        "MAE and RMSE measure prediction error; lower is better. "
        "R² measures how well the model explains variation "
        "in the test-set target values."
    )

    st.subheader("Compare Model Errors")

    comparison_data = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Linear Regression",
            "Random Forest",
            "Random Forest"
        ],
        "Metric": ["MAE", "RMSE", "MAE", "RMSE"],
        "Error": [53.32, 74.56, 32.75, 50.53]
    })

    fig = px.bar(
        comparison_data,
        x="Metric",
        y="Error",
        color="Model",
        barmode="group",
        color_discrete_map={
            "Linear Regression": "#38BDF8",
            "Random Forest": "#65A30D"
        },
        labels={
            "Error": "Error (thousand dollars)",
            "Metric": "Evaluation Metric"
        },
        title="Model Error Comparison"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#222222"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.write(
        "Random Forest performed better on this test split. "
        "This does not guarantee the same results on new data."
    )


# --------------------------------
# PAGE 3: FEATURE IMPORTANCE
# --------------------------------
elif page == "Feature Importance":

    st.title("Feature Importance")

    st.write(
        "This chart shows the relative importance assigned "
        "to each feature by the Random Forest model."
    )

    importance_data = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_data = importance_data.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        importance_data,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale=["#BAE6FD", "#4D7C0F"],
        title="Importance of Housing Features"
    )

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="#222222",
        coloraxis_showscale=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Feature importance does not prove that a feature "
        "causes house prices to change."
    )

    st.dataframe(
        importance_data.sort_values(
            by="Importance",
            ascending=False
        ),
        use_container_width=True,
        hide_index=True
    )


# Footer
st.divider()

st.caption(
    "House Price Predictor | Built with Python, Pandas, "
    "Scikit-learn and Streamlit"
)
