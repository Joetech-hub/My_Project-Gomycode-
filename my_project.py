# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 13:17:51 2024

@author: PC
"""


# import all necessary libraries
import pandas as pd
import streamlit as st

from joblib import load

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from PIL import Image


# load model data
life_expectant_model = load('world_life_expectancy.joblib')



# Encode Categorical Variables

# create a function that preprocesses input data

def cat_encoder(df):
    le = LabelEncoder()
    df['Country'] = le.fit_transform(df['Country'])
    df['Status'] = le.fit_transform(df['Status'])
    return df
def normaliser(df):
    mms = MinMaxScaler()
    numerical_cols = ['Adult Mortality','Infant death','Alcohol','percentage expenditure','Hepatitis B','Measles', 'BMI', 'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'Schooling']
    df[numerical_cols] = mms.fit_transform(df[numerical_cols])
    return df
# create a function that preprocesses input data
def preprocessor(inp_df):
    user_df = cat_encoder(inp_df)
    user_df = normaliser(user_df)
    return user_df
# Creating web app interface
def main():
    st.title('Average Life Expectancy Predictor')
    st.write('This App predicts the average life expectancy in a country given some health and economic indicators ')
    img = Image.open('world-life-expectancy.png')
    st.image(img, width=500)
    
    # Create dictionary to store input data
    input_data = {}
    col1, col2, col3 = st.columns(3) # split the interface into 3 columns
    with col1:
        input_data['Country'] = st.selectbox('Country', ['Afghanistan', 'Albania', 'Algeria', 'Angola', 'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan', 'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina', 'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria', 'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde', 'Cambodia', 'Cameroon', 'Canada', 'Central African Republic', 'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo', 'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus', 'Czechia', "Democratic People's Republic of Korea", 'Democratic Republic of the Congo', 'Denmark', 'Djibouti', 'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia', 'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia', 'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala', 'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'Iceland', 'India', 'Indonesia', 'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati', 'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic", 'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania', 'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives', 'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius', 'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia', 'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia', 'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau', 'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Qatar', 'Republic of Korea', 'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda', 'Saint Kitts and Nevis', 'Saint Lucia', 'Saint Vincent and the Grenadines', 'Samoa', 'San Marino', 'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia', 'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia', 'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden', 'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand', 'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo', 'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey', 'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine', 'United Arab Emirates', 'United Kingdom of Great Britain and Northern Ireland', 'United Republic of Tanzania', 'United States of America', 'Uruguay', 'Uzbekistan', 'Vanuatu', 'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen', 'Zambia', 'Zimbabwe'])
        input_data['Year'] = st.selectbox('Year', [2015, 2014, 2013, 2012, 2011, 2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000])
        input_data['Status'] = st.selectbox('The Development status of your Country', ['Developing', 'Developed'])
        input_data['Adult Mortality'] = st.slider('Gauge the probabilities of survival between ages 15 and 60 per 1,000 population in your country', min_value=1.0, max_value=173.0, step=1.0)
        input_data['Infant death'] = st.slider('What is the number of infant deaths per 1,000 live births in your country', min_value=0.0, max_value=1800.0, step=100.0)
        input_data['Alcohol'] = st.slider('Average alcohol consumption in liters per capital', min_value=0.01, max_value=17.87, step=0.01)
    
    with col2:
        input_data['percentage expenditure'] = st.slider("Health expenditure as a percentage of a country's GDP", min_value= 0.0, max_value=19480.0, step=500.0)
        input_data['Hepatitis B'] = st.slider('Percentage of immunization coverage for Hepatitis B.', min_value=1.0, max_value=99.0, step=1.0)
        input_data['Measles'] = st.slider('The number of reported cases of Measles per 1,000 population', min_value= 0.0, max_value= 212183.0, step= 1000.0)
        input_data['BMI'] = st.slider('Average Body Mass Index', min_value= 1.0, max_value= 87.3, step= 1.0)
        input_data['Polio'] = st.slider('Percentage of immunization coverage for Polio', min_value= 3.0 , max_value= 99.0, step= 1.0)
        input_data['Total expenditure'] = st.slider('Total health expenditure as a percentage of GDP', min_value= 0.37, max_value= 17.6, step= 1.0)
        
    with col3:
        input_data['Diphtheria'] = st.slider('Percentage of immunization coverage for Diphtheria.', min_value= 2.0, max_value= 99.0, step= 1.0)
        input_data['HIV/AIDS'] = st.slider('Prevalence of HIV/AIDS as a percentage of the population', min_value= 0.1, max_value= 50.6, step= 1.0)
        input_data['GDP'] = st.slider('Gross Domestic Product', min_value= 1.68, max_value= 119172.74, step= 1000.0)
        input_data['Population'] = st.slider("Nation's population", min_value= 34.0, max_value= 1293859294.0, step= 100000.0)
        input_data['Schooling'] = st.slider('Average years of schooling', min_value= 0.0, max_value= 20.7, step= 1.0)
    
    input_df = pd.DataFrame(input_data, index=[0])  # Convert input data to DataFrame
    st.write(input_df)  # Display the collected data on the app interface
    
    if st.button('PREDICT'):  # When the 'Predict' button is clicked
        final_df = preprocessor(input_df)  # Preprocess the collected data
        prediction = life_expectant_model.predict(final_df)[0].round(1)  # Use the model to predict the outcome
        # Display the prediction result
        st.subheader("The Average Life Expectancy In Your Country is: {} Years".format(prediction))

# Run the main function when the script is executed directly
if __name__ == '__main__':
    main()
