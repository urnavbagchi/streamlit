import pandas as pd
import numpy as np
import streamlit as st
import joblib

@st.cache_data
def load_model():
    model_path = "models/random_forest_1.joblib"
    return joblib.load(model_path)

def convert_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Create Streamlit containers
header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

# Define Streamlit sections
with header:
    st.title('Talent Analytics Tool')
    st.text('This is a simple app to test the model framework')

with dataset:
    st.header('Bulk upload of data')
    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is not None:
        rfc = load_model()
        df_test = pd.read_csv(uploaded_file)
        df_candidate = df_test['candidate_full_name']
        df_location = df_test['location']
        df_skills = df_test['skills']
        df_test.drop(['candidate_full_name', 'location', 'skills'], axis=1, inplace=True)
        df_test.industry[df_test.industry == 'Financial Service'] = 1
        df_test.industry[df_test.industry == 'Internet'] = 2
        df_test.industry[df_test.industry == 'IT Services'] = 3
        df_test.industry[df_test.industry == 'Other'] = 4
        df_test.industry[df_test.industry == 'Software Development'] = 5  
        df_test.industry[df_test.industry == 'Electronics & Telecomm'] = 6 

        df_test.total_experience[df_test.total_experience == 'Less than 10 years'] = 4
        df_test.total_experience[df_test.total_experience == '10 + years'] = 5

        predict_test_proba = rfc.predict_proba(df_test)
        df_test.insert(0, 'candidate_full_name', df_candidate)
        df_test.insert(4, 'location', df_location)
        df_test.insert(5, 'skills', df_skills)
        df_test['selection_proba'] = predict_test_proba[:,1]
        to_percent = lambda x: f'{x:.2%}'
        df_test['selection_proba'] = df_test['selection_proba'].apply(to_percent)

        df_test.industry[df_test.industry == 1] = "Financial Service"
        df_test.industry[df_test.industry == 2] = "Internet"
        df_test.industry[df_test.industry == 3] = "IT Services"
        df_test.industry[df_test.industry == 4] = "Other"
        df_test.industry[df_test.industry == 5] = "Software Development"  
        df_test.industry[df_test.industry == 6] = "Electronics & Telecomm" 

        df_test.total_experience[df_test.total_experience == 4] = "Less than 10 years"
        df_test.total_experience[df_test.total_experience == 5] = "10 + years"

        st.write(df_test)
        csv = convert_to_csv(df_test)
        download1 = st.download_button(label="Download data as CSV", data=csv, file_name='Prediction.csv', mime='text/csv')

with model_training:
    st.header('Model Training')
    st.text('Please fill in the information below')
    sel_col, sel_col2, sel_col3 = st.columns(3)

    university_tier = sel_col.selectbox('University Tier', ['1', '2'])
    industry  = sel_col.selectbox('Industry', ['Financial Service', 'Internet', 'IT Services', 'Software Development', 'Electronics & Telecomm', 'Other'])
    total_experience = sel_col.slider('Total Experience', 0, 20, 1)
    location = sel_col2.selectbox('Location', ['Delhi', 'Mumbai', 'Chennai', 'Bangalore', 'Hyderabad', 'Pune', 'Other'])
    skills = sel_col2.selectbox('Skills', ['AI/ML', 'Android', 'Big Data','C/C++/Java/Python', 'Cloud', 'Full Stack', 'IOT/Blockchain', 'Network/Security'])
    btnResult = sel_col.button('Predict')

    if industry == 'Financial Service':
        industry_code = 1
    elif industry == 'Internet':
        industry_code = 2
    elif industry == 'IT Services':
        industry_code = 3
    elif industry == 'Software Development':
        industry_code = 5
    elif industry == 'Electronics & Telecomm':
        industry_code = 6
    elif industry == 'Other':
        industry_code = 4
    
    if total_experience < 10:
        experience_code = 4
    else:
        experience_code = 5

    rfc = load_model()
    df = pd.DataFrame([[industry_code, university_tier, experience_code]], columns=['industry', 'university_tier', 'total_experience'])
    predict_proba = rfc.predict_proba(df)
    to_percent = lambda x: f'{x:.2%}'
    df_new = pd.DataFrame({'Probability': predict_proba[:,1]})
    df_new = df_new.applymap(to_percent)

    if btnResult:
        sel_col3.write('Based on the information provided, the probability of selection is: ' + df_new['Probability'].iloc[0])
