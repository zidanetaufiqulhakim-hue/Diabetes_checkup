import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import joblib
import pickle 
import os


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


# Create two columns for layout
c1, c2 = st.columns(2)

# Create Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", options=["Home","About The Model)"])

if page == "Home":
    st.title("Quick Diabetes Check-Up")
    st.write("*What the app dos (think later)")
    with c1:

        st.header("Estimate Your Diabetes Risk Here!")
        # Gether Basic Information
        gender = st.selectbox("Gender", options=["Male", "Female"])
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        weight = st.number_input("Weight (in kg)", min_value=0.0, max_value=300.0, value=70.0)
        height = st.number_input("Height (in cm)", min_value=0.0,  max_value=250.0, value=170.0)
        bmi = weight / ((height / 100) ** 2) # Calculate BMI from weight and height

        # Gether Lifestyle Information
        family_history_diabetes = st.selectbox("Family History of Diabetes", options=["Yes", "No"])
        physical_activity_minutes_per_week = st.number_input("Physical Activity (minutes per week)", min_value=0, max_value=10000, value=150)
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
        if st.button("Estimate Diabetes Risk"):
            risk_score = predict_diabetes_risk(input_data)

            # Display the risk score
            st.subheader("Estimated Diabetes Risk Score:")
            st.markdown(f"{risk_score[0]:.0f}")
            # Display risk interpretationstreamlit
            if risk_score <= 22:
                st.success("Low Risk of Diabetes")
            elif 23 <= risk_score <= 40:
                st.warning("Moderate Risk of Diabetes")
            else:
                st.error("High Risk of Diabetes")

    with c2:
        st.header("Check Diabetes Status Here!")
        glucose_postprandial = st.number_input("Enter Postprandial Glucose Level (mg/dL) 3 Hours After Meal", min_value=70, max_value=300)
        risk_score = st.number_input("Re-enter Your Diabetes Risk Score", min_value=0, max_value=100)

        # Prepare input data for diabetes diagnoser
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

elif page == "About The Model":
    st.title("About The Model")