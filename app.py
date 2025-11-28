import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
import pickle 
import os

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

def diet_score_checkbox():

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
st.sidebar.title("Go To")

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

#-------------------------------- RENDERING PAGES --------------------------------#

# -------- Page: Estimate Diabtes Risk ---------# 

if st.session_state.Page == "Estiiamate Diabetes Risk":
    st.title("Diabetes Risk Estimator")
    st.write("This app estimates your risk of developing diabetes based on your personal and lifestyle information.")

    # Enter Basic Information
    st.header("1. Enter Your Basic Information:")

    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (in cm)", min_value=0.0,  max_value=250.0, value=170.0)
    bmi = weight / ((height / 100) ** 2) # Calculate BMI from weight and height

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
    diet_score = diet_score_checkbox()

    # How many hours do you sleep per day?
    st.subheader("How many hours do you sleep per day?")
    sleep_hours_per_day = st.number_input("Sleep Duration",min_value=0, max_value=12, value=7)

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
    
    if st.button("Estimate Diabetes Risk"):
        risk_score = predict_diabetes_risk(input_data)
        st.markdown("Estimated Diabetes Risk Score:")
        c1, c2 = st.columns(2)
        with c1:
            # Display the risk scorestreamli
            st.header(f"{risk_score[0]:.0f}")

        with c2:
            # Display risk interpretationstreamlit
                if risk_score < 28:
                    st.success("Low Risk of Diabetes")
                elif 28 <= risk_score <= 40:
                    st.warning("Moderate Risk of Diabetes")
                elif risk_score > 40:
                    st.error("High Risk of Diabetes")
                    
                st.write(" Below 25: Low Risk | 26-40: Moderate Risk | Above 40: High Risk ")


# -------- Page: Diabetes Diagnoser ---------#
if st.session_state.Page == "Diabetes Diagnoser":
    st.header("Check Diabetes Status Here!")
    glucose_postprandial = st.number_input("Enter Postprandial Glucose Level (mg/dL) 3 Hours After Meal", min_value=70, max_value=300)

    # ---Enter Basic Information---
    st.header("1. Enter Your Basic Information:")
    gender = st.selectbox("Gender", options=["Male", "Female"])
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
    height = st.number_input("Height (in cm)", min_value=0.0,  max_value=250.0, value=170.0)
    bmi = weight / ((height / 100) ** 2) # Calculate BMI from weight and height

    # ---Gether Lifestyle Information---
    st.header("2. Tell Me about Your Lifestyle:")

    # is there any family history of diabetes?
    st.subheader("Is there any family history of diabetes?")
    family_history_diabetes = st.selectbox(options=["Yes", "No"])

    # # How often do you exercise per week?
    st.subheader("How often do you exercise per week?")
    physical_activity_minutes_per_week = st.radio()
    diet_score = st.slider("Diet Score (1-10)", min_value=1, max_value=10, value=5)
    sleep_hours_per_day = st.number_input("Sleep Hours (hours per day)", min_value=0, max_value=12, value=7)

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

            # Display diabetes statusstream
            st.markdown("Diabetes Diagnosis Result:")
            if diabetes_status[0] == "No Diabetes":
                st.markdown("HEALTHY!")
                st.success("You are not diagnosed with Diabetes.")

            elif diabetes_status[0] == "Pre-Diabetes":
                st.markdown("You are diagnosed with Pre-Diabetess")
                st.warning("Please consult a healthcare professional for further evaluation.")

            else:
                st.markdown("You are diagnosed with Diabetes")
                st.error("Please consult a healthcare professional for proper management and treatment.")