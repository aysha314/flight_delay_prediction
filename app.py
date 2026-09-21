import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml.pipeline import PipelineModel
from pyspark.ml.classification import RandomForestClassificationModel


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Flight Delay Prediction",
    page_icon="✈️",
    layout="wide"
)


# ============================================================
# CREATE SPARK SESSION
# ============================================================

@st.cache_resource
def create_spark():

    spark = (
        SparkSession.builder
        .appName("Flight Delay Prediction")
        .master("local[*]")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    return spark


spark = create_spark()


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    feature_pipeline = PipelineModel.load(
        "models/feature_pipeline_model"
    )

    random_forest = RandomForestClassificationModel.load(
        "models/final_random_forest_model"
    )

    return feature_pipeline, random_forest


feature_pipeline, random_forest = load_models()


# ============================================================
# GET VALID VALUES FROM TRAINED PIPELINE
# ============================================================

airline_options = feature_pipeline.stages[0].labels
origin_options = feature_pipeline.stages[1].labels
destination_options = feature_pipeline.stages[2].labels


# ============================================================
# HEADER
# ============================================================

st.title("✈️ Flight Delay Prediction System")

st.markdown(
    """
    ### Predict whether a flight is likely to be delayed

    Enter the flight details below and the trained
    Random Forest model will generate a prediction.
    """
)

st.divider()


# ============================================================
# FLIGHT INFORMATION
# ============================================================

st.subheader("🛫 Flight Information")

col1, col2 = st.columns(2)


with col1:

    airline = st.selectbox(
        "Airline",
        airline_options
    )

    origin = st.selectbox(
        "Origin Airport",
        origin_options
    )

    destination = st.selectbox(
        "Destination Airport",
        destination_options
    )

    month = st.selectbox(
        "Month",
        list(range(1, 13))
    )


with col2:

    day_of_week = st.selectbox(
        "Day of Week",
        list(range(1, 8))
    )

    departure_hour = st.slider(
        "Departure Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    distance = st.number_input(
        "Distance (miles)",
        min_value=0.0,
        value=500.0,
        step=1.0
    )

    crs_elapsed_time = st.number_input(
        "Scheduled Flight Duration (minutes)",
        min_value=1.0,
        value=120.0,
        step=1.0
    )


st.divider()


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Flight Delay",
    use_container_width=True
)


if predict_button:

    # --------------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------------

    flight_data = spark.createDataFrame([
        Row(
            op_unique_carrier=airline,
            origin=origin,
            dest=destination,
            month=int(month),
            day_of_week=int(day_of_week),
            departure_hour=int(departure_hour),
            distance=float(distance),
            crs_elapsed_time=float(crs_elapsed_time)
        )
    ])


    # --------------------------------------------------------
    # Apply feature pipeline
    # --------------------------------------------------------

    prepared_flight = feature_pipeline.transform(
        flight_data
    )


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction_result = random_forest.transform(
        prepared_flight
    )


    result = prediction_result.select(
        "prediction",
        "probability"
    ).collect()[0]


    prediction = int(result["prediction"])

    delay_probability = float(
        result["probability"][1]
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ The flight is predicted to be DELAYED"
        )

    else:

        st.success(
            "✅ The flight is predicted to be NOT DELAYED"
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.metric(
        "Delay Probability",
        f"{delay_probability * 100:.2f}%"
    )


    st.progress(delay_probability)


    # ========================================================
    # FLIGHT SUMMARY
    # ========================================================

    st.subheader("✈️ Flight Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    summary_col1.metric(
        "Airline",
        airline
    )

    summary_col2.metric(
        "Route",
        f"{origin} → {destination}"
    )

    summary_col3.metric(
        "Departure Hour",
        f"{departure_hour}:00"
    )