import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
import pickle 
import os

desclaimer = """⚠ **Disclaimer:**  
This application uses **synthetic (artificially generated) health data that reflect real data from Kaggle.** Use this App for **educational, research, and self-screening purposes only**.  
It is **not** intended to diagnose, treat, or replace professional medical evaluation.  
For any health concerns, please consult a licensed healthcare provider."""

#-------------------------------- FUNCTION DEFINITIONS --------------------------------#

def predict_diabetes_risk(input_data):
        with open('diabetes-risk-estimator-model.pkl', 'rb') as f:
            db_risk_estimator = joblib.load(f)

        # Convert input data to DataFrame
        input_df = pd.DataFrame([input_data], columns=input_data.keys())

        # Predict diabetes risk using the loaded model
        risk_score = db_risk_estimator.predict(input_df)
        return risk_score

def diagnose_diabetes(input_data_db_diagnoser):
    # Load the pre-trained diabetes classifier model
    with open('diabetes-classifier.pkl', 'rb') as f:
        db_classifier = pickle.load(f)

    # Convert input data to DataFrame
    input_df_db_diagnoser = pd.DataFrame([input_data_db_diagnoser], columns=input_data_db_diagnoser.keys())

    #diagose user
    diabetes_status = db_classifier.predict(input_df_db_diagnoser)
    return diabetes_status

def categorize_age(age):
        if age < 12:
            return "childern"
        elif 12 <= age < 18:
            return "teens"
        elif 18 <= age < 30:
            return "young_adult"
        elif 30 <= age < 45:
            return "early_mid-age"
        elif 45 <= age < 60:
            return "old_mid-age"
        else:
            return "senior"

def categorize_diet_score(diet_score):
    if diet_score <= 3:
        return "very_poor"
    elif 4 <= diet_score <= 5:
        return "poor"
    elif 6 <= diet_score <= 7:
        return "avarage"
    else:
        return "good"   
    
def sleep_quality_category(sleep_hours):
    if sleep_hours < 6:
        return "poor"
    elif 6 <= sleep_hours <= 7:
        return "suboptimal"
    elif 8 <= sleep_hours <= 9:
        return "optimal"
    else:
        return "long_sleep"

def diet_score_radio():

    diet_score = st.radio( "Diet Quality Score",
    options=[3, 5, 7, 9],
    format_func=lambda x: {
        3: "VERY POOR: Frequent sugary drinks, high-carb snacks, low fibre, irregular meals.",
        5: "POOR: Still consumes sweets, fried foods, refined carbs. Low vegetables.",
        7: "AVARAGE: Sometimes healthy, sometimes not. Moderate carbs but inconsistent habits.",
        9: "GOOD: Balanced meals, high fibre, lean protein, low GI carbs."
    }[x])

    return diet_score

def physical_activity_radio():
    physical_activity_minutes_per_week = st.radio("Physical Activity Level",
    options=[50, 100, 150, 200],
    format_func=lambda x: {
        50: "Sedentary Lifestyle: Little to no exercise, mostly sitting or lying down.",
        100: "Light Activity: Light exercise or sports 1-2 days a week.",
        150: "Moderate Activity: Moderate exercise or sports 3-5 days a week.",
        200: "High Activity: Hard exercise or sports 6-7 days a week."
    }[x])
    return physical_activity_minutes_per_week
    
#-------------------------------- SETTING UP THE APP --------------------------------#
# Set Default Page

if "Page" not in st.session_state:
    st.session_state.Page = "Estiiamate Diabetes Risk"
else:
    pass

# --------Side Bar Go-to---------
st.sidebar.title("🩺 ")
st.sidebar.header("Go to:")

# Container 1
with st.sidebar.container():
    if st.button("Estimate Diabetes Risk"):
        st.session_state.Page = "Estiiamate Diabetes Risk"

# Container 2
with st.sidebar.container():
    if st.button("Diabetes Diagnoser"):
        st.session_state.Page = "Diabetes Diagnoser"

#container 3
with st.sidebar.container():
    if st.button("About The Model"):
        st.session_state.Page = "About The Model"

# copyright info
st.sidebar.divider()
st.sidebar.markdown("© 2025 zidanetaufiqulhakim-hue (GitHub). All rights reserved.")
# repository link for contributions and learning
st.sidebar.divider()
st.sidebar.markdown("Contributions and feedback are welcome!")
st.sidebar.markdown("Repository:")
st.sidebar.write("https://github.com/zidanetaufiqulhakim-hue/Diabetes_checkup")

#-------------------------------- RENDERING PAGES --------------------------------#

# -------- Page: Estimate Diabtes Risk ---------# 

if st.session_state.Page == "Estiiamate Diabetes Risk":
    st.title("Diabetes Risk Estimator")
    st.write("Welcome to the Diabetes Risk Estimator! 🩺 This app helps you assess your risk of diabetes based on your personal information, lifestyle habits, and blood sugar levels.")
    
    # Instructions to use the app
    st.markdown("## How to use:")
    st.markdown("1. Fill in your **basic information**: gender, age, weight, and height")
    st.markdown("2. **Tell me about your lifestyle**: family history of diabetes, exercise frequency, diet quality, and sleep hours.")
    st.markdown("3. **Click** the **'Estimate Diabetes Risk'** button to get your risk score and interpretation.")

    # note to users
    st.info(desclaimer)
    st.divider()

    # Enter Basic Information
    st.header("1. Enter Your Basic Information:")

    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (in cm)", min_value=0.0,  max_value=250.0, value=170.0)
    bmi = weight / ((height / 100) ** 2) # Calculate BMI from weight and height
    st.divider()

    # ---Gether Lifestyle Information---
    st.header("2. Tell Me about Your Lifestyle:")

    # is there any family history of diabetes?
    st.subheader("Is there any family history of diabetes?")
    family_history_diabetes = st.selectbox("Famaily Diabetes History",options=["Yes", "No"])

    # How often do you exercise per week?
    st.subheader("How often do you exercise per week?")
    physical_activity_minutes_per_week = physical_activity_radio()

    # wha best describes your diet?
    st.subheader("What best describes your diet?")
    diet_score = diet_score_radio()

    # How many hours do you sleep per day?
    st.subheader("How many hours do you sleep per day?")
    sleep_hours_per_day = st.number_input("Sleep Duration",min_value=0, max_value=12, value=7)
    st.divider()

    # Feature Engineering the input data
    age_bmi = age * bmi
    health_score = (sleep_hours_per_day + physical_activity_minutes_per_week + diet_score) / 3
    age_category = categorize_age(age)
    diet_score_category = categorize_diet_score(diet_score)
    sleep_quality = sleep_quality_category(sleep_hours_per_day)

    # Prepare input data for prediction
    input_data = {
        "gender": gender,
        "family_history_diabetes": family_history_diabetes,
        "age*bmi": age_bmi,
        "health_score": health_score,
        "age_category": age_category,
        "sleep_quality": sleep_quality,
        "diet_score_category": diet_score_category
        }

    # Predict diabetes risk
    st.header("3. Get Your Diabetes Risk Estimate:")
    if st.button("Estimate Diabetes Risk"):
        risk_score = predict_diabetes_risk(input_data)
    
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Estimated Diabetes Risk Score:**")
            # Display the risk scorestreamlit
            c1a, c1b, c1c = st.columns(3)
            with c1b: # -> center the output
                st.markdown(f"# {risk_score[0]:.0f}")

        with c2:
                # Display risk categories
                st.markdown(" *Below 26: Low Risk | 26-40: Moderate Risk | Above 40: High Risk*")

                # Display risk interpretationstreamlit
                if risk_score < 28:
                    st.success("Low Risk of Diabetes")
                elif 28 <= risk_score <= 40:
                    st.warning("Moderate Risk of Diabetes")
                elif risk_score > 40:
                    st.error("High Risk of Diabetes")



# -------- Page: Diabetes Diagnoser ---------#
elif st.session_state.Page == "Diabetes Diagnoser":
    st.title("Check Diabetes Status Here!")
    st.write(
        "Welcome to the Diabetes Risk Checker! 🩺 This app helps you assess your diabetes based on your personal information, lifestyle habits, and blood sugar levels."
    )

    # Instructions to use the app
    st.markdown("## How to use:")
    st.markdown("1. Fill in your **basic information**: gender, age, weight, and height")
    st.markdown("2. **Tell me about your lifestyle**: family history of diabetes, exercise frequency, diet quality, and sleep hours.")
    st.markdown("3. Provide your **Postprandial Glucose Level** (3 hours after meal).")
    st.markdown("4. **Click** the **'Diagnose Diabetes'** button to get your diabetes status.")

    # Note to users
    
    st.info(desclaimer)
    
    st.divider()
    
    # ---Enter Basic Information---
    st.header("1. Enter Your Basic Information:")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (in cm)", min_value=0.0,  max_value=250.0, value=170.0)
    bmi = weight / ((height / 100) ** 2) # Calculate BMI from weight and height
    st.divider()

    # ---Gether Lifestyle Information---
    st.header("2. Tell Me about Your Lifestyle:")

    # is there any family history of diabetes?
    st.subheader("Is there any family history of diabetes?")
    family_history_diabetes = st.selectbox("Famaily Diabetes History",options=["Yes", "No"])

    # # How often do you exercise per week?
    st.subheader("How often do you exercise per week?")
    physical_activity_minutes_per_week = physical_activity_radio()

    # wha best describes your diet?
    st.subheader("What best describes your diet?")
    diet_score = diet_score_radio()
    sleep_hours_per_day = st.number_input("Sleep Hours (hours per day)", min_value=0, max_value=12, value=7)
    st.divider()

    # ----Gather Postprandial Glucose Level After 3 hours of Meal----
    st.header("3. After 3 hours of meal, how much is your Postprandial Glucose Level?")
    glucose_postprandial = st.number_input("Enter Postprandial Glucose Level (mg/dL) 3 Hours After Meal", min_value=70, max_value=300)

    # Feature Engineering the input data
    age_bmi = age * bmi
    health_score = (sleep_hours_per_day + physical_activity_minutes_per_week + diet_score) / 3
    age_category = categorize_age(age)
    diet_score_category = categorize_diet_score(diet_score)
    sleep_quality = sleep_quality_category(sleep_hours_per_day)

    # Prepare input data for prediction
    input_data = {
        "gender": gender,
        "family_history_diabetes": family_history_diabetes,
        "age*bmi": age_bmi,
        "health_score": health_score,
        "age_category": age_category,
        "sleep_quality": sleep_quality,
        "diet_score_category": diet_score_category
        }
    
    # Predict diabetes risk before diagnosing then use it as input to diagnoser
    risk_score = predict_diabetes_risk(input_data)

    # Prepare input data for diabetes diagnoserstreamlit run
    input_data_db_diagnoser = {
    "diabetes_risk_score": risk_score,
    "glucose_postprandial": glucose_postprandial
    }
    # Diagnose diabetes

    if st.button("Diagnose Diabetes"):

        if glucose_postprandial == None or glucose_postprandial <= 0:
            st.error("Please enter a valid Postprandial Glucose Level")
        else:
            diabetes_status = diagnose_diabetes(input_data_db_diagnoser)

            # Display diabetes status
            st.markdown("Diabetes Diagnosis Result:")
            if diabetes_status[0] == "No Diabetes":
                st.header("HEALTHY!")
                st.success("You are not diagnosed with Diabetes.")

            elif diabetes_status[0] == "Pre-Diabetes":
                st.header("Pre-Diabetess")
                st.warning("Please consult a healthcare professional for further evaluation.")

            else:
                st.header("Type 2 Diabetes")
                st.error("Please consult a healthcare professional for proper management and treatment.")
    

# -------- Page: About The Model ---------#
elif st.session_state.Page == "About The Model":
    st.title("📘 About the Models")

    st.markdown("""
    This app uses **two machine learning models** to help you understand your diabetes status:

    1. **Diabetes Risk Estimator** → predicts your *numerical* diabetes risk score.  
    2. **Diabetes Diagnoser** → predicts whether you're **No Diabetes**, **Pre-Diabetes**, or **Type 2 Diabetes** based on your inputs.
    """)

    st.divider()

    # ------------------------------
    # DIABETES RISK ESTIMATOR
    # ------------------------------
    st.header("Diabetes Risk Estimator (Gradient Boosting Regressor)")

    st.markdown("""
    The **Diabetes Risk Estimator** is built using a **Gradient Boosting Regressor**,  
    a model that combines many small decision trees to make a strong predictor.

    ###Model Performance
    - **MSE (Mean Squared Error):** `2.2601`  
    - **MAE (Mean Absolute Error):** `1.2084`  

    ###What These Metrics Mean:
    - **MSE** measures how far predictions are from actual values **on average, squared**.  
      - Lower = better.  
      - An MSE of *2.26* means the model’s predictions are reasonably close to real risk scores.
    - **MAE** measures the **average absolute difference** between predicted and actual scores.  
      - An MAE of *1.20* means predictions are usually within **±1.2 risk points** of the true value.  
      - This makes the estimator reliable for **screening**, not medical diagnosis.

    ### Trained Features
    - **Categorical Features:**  
      `gender`, `family_history_diabetes`
    - **Ordinal Features:**  
      `age_category`, `sleep_quality`, `diet_score_category`
    - **Numerical Features:**  
      `health_score`, `age*bmi`
    """)

    st.divider()

    # ------------------------------
    # DIABETES DIAGNOSER
    # ------------------------------
    st.header("Diabetes Diagnoser (Logistic Regression)")

    st.markdown("""
    The Diabetes Diagnoser uses a **Logistic Regression classifier**,  
    a simple and highly interpretable model commonly used in medical research.

    It predicts:
    - **No Diabetes**
    - **Pre-Diabetes**
    - **Type 2 Diabetes**

    ###  Model Performance
    | Class | Precision | Recall | F1-Score | Support |
    |-------|-----------|--------|----------|----------|
    | No Diabetes | 0.78 | 0.78 | 0.78 | 1406 |
    | Pre-Diabetes | 0.63 | 0.63 | 0.63 | 1394 |
    | Type 2 | 0.82 | 0.82 | 0.82 | 1400 |
    | **Accuracy** | **0.74** | — | — | 4200 |
    | **Macro Avg** | **0.74** | **0.74** | **0.74** | — |
    | **Weighted Avg** | **0.74** | **0.74** | **0.74** | — |

    ### Metric Interpretation (Simplified for Users)

    - **Precision** → When the model predicts a class, how *correct* is it?  
      Higher precision means fewer **false alarms**.
    - **Recall** → Of all real cases, how many did the model successfully detect?  
      Higher recall means fewer **missed cases**.
    - **F1-Score** → Balance of precision + recall.  
      Good for health-related predictions.
    - **Accuracy (0.74)** → The model correctly predicts 74% of all cases.

    ✔ The model is strongest at detecting **Type 2 Diabetes**.  
    ✔ It performs well on **No Diabetes**.  
    ⚠ It is moderately accurate for **Pre-Diabetes**, which is the hardest class to predict.

    ### Trained Features
    - `postprandial_glucose_level (3 hours after meal)`
    - `diabetes_risk_score` (output from the previous model)
    """)

    st.divider()

    st.write("Trainning Dataset https://www.kaggle.com/datasets/mohankrishnathalla/diabetes-health-indicators-dataset")

    st.info(desclaimer)
