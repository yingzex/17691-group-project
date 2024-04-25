import streamlit as st
import pandas as pd
import joblib

# Load your model (adjust the file path as necessary)
# Load your model (adjust the file path as necessary)
model = joblib.load('ran_model.pkl')

# If the model is a pipeline, you can inspect steps to get feature names
if hasattr(model, 'named_steps'):
    # Assuming there's a step in the pipeline that involves feature transformation
    # and that it stores feature names. This could be a ColumnTransformer or similar.
    transformer = model.named_steps['name_of_transformer_step']
    if hasattr(transformer, 'get_feature_names'):
        feature_names = transformer.get_feature_names()
        print("Model expects the following features:", feature_names)
    else:
        print("Transformer does not support get_feature_names().")
else:
    print("Loaded model is not a pipeline, or the specific step is not found.")

# App title
st.title('COVID-19 Data Prediction Model')

# Creating form for user input
with st.form("input_form"):
    weekly_cases = st.number_input('Weekly Cases', value=0.0, format="%.1f")
    year = st.number_input('Year', min_value=2019, max_value=2030, value=2020, format="%d")
    weekly_cases_per_million = st.number_input('Weekly Cases per Million', value=0.0, format="%.3f")
    weekly_deaths = st.number_input('Weekly Deaths', value=0.0, format="%.1f")
    weekly_deaths_per_million = st.number_input('Weekly Deaths per Million', value=0.0, format="%.3f")
    total_vaccinations = st.number_input('Total Vaccinations', value=0.0)
    people_vaccinated = st.number_input('People Vaccinated', value=0.0)
    people_fully_vaccinated = st.number_input('People Fully Vaccinated', value=0.0)
    total_boosters = st.number_input('Total Boosters', value=0.0)
    daily_vaccinations = st.number_input('Daily Vaccinations', value=0.0)
    total_vaccinations_per_hundred = st.number_input('Total Vaccinations per Hundred', value=0.0, format="%.3f")
    people_vaccinated_per_hundred = st.number_input('People Vaccinated per Hundred', value=0.0, format="%.3f")
    people_fully_vaccinated_per_hundred = st.number_input('People Fully Vaccinated per Hundred', value=0.0, format="%.3f")
    total_boosters_per_hundred = st.number_input('Total Boosters per Hundred', value=0.0, format="%.3f")
    daily_vaccinations_per_hundred = st.number_input('Daily Vaccinations per Hundred', value=0.0, format="%.3f")
    daily_people_vaccinated = st.number_input('Daily People Vaccinated', value=0.0)
    daily_people_vaccinated_per_hundred = st.number_input('Daily People Vaccinated per Hundred', value=0.0, format="%.3f")
 

    continent = st.selectbox('Continent', options=['Africa', 'Asia', 'Europe', 'North America', 'Oceania', 'South America'])

    # Handling categorical continent data
    continent_data = {
        'Continent_Africa': 0, 'Continent_Asia': 0, 'Continent_Europe': 0,
        'Continent_North America': 0, 'Continent_Oceania': 0, 'Continent_South America': 0
    }
    continent_data[f'Continent_{continent}'] = 1

    submit_button = st.form_submit_button("Predict")

# Prediction function
def predict(input_data):
    df = pd.DataFrame([input_data], columns=[
        'Weekly Cases', 'Year', 'Weekly Cases per Million', 'Weekly Deaths',
        'Weekly Deaths per Million', 'Total Vaccinations', 'People Vaccinated',
        'People Fully Vaccinated', 'Total Boosters', 'Daily Vaccinations',
        'Total Vaccinations per Hundred', 'People Vaccinated per Hundred',
        'People Fully Vaccinated per Hundred', 'Total Boosters per Hundred',
        'Daily Vaccinations per Hundred', 'Daily People Vaccinated',
        'Daily People Vaccinated per Hundred', 'Continent_Africa', 'Continent_Asia',
        'Continent_Europe', 'Continent_North America', 'Continent_Oceania',
        'Continent_South America'
    ])
    expected_columns = {'Weekly Cases', 'Year', 'Weekly Cases per Million', 'Weekly Deaths', 'Weekly Deaths per Million',
                    'Total Vaccinations', 'People Vaccinated', 'People Fully Vaccinated', 'Total Boosters', 'Daily Vaccinations'}

    # Check before model training or prediction
    assert expected_columns <= set(df.columns), "Missing columns in the dataset"

    prediction = model.predict(df)
    return prediction

# Display prediction result
if submit_button:
    input_data = [
        weekly_cases, year, weekly_cases_per_million, weekly_deaths,
        weekly_deaths_per_million, total_vaccinations, people_vaccinated,
        people_fully_vaccinated, total_boosters, daily_vaccinations,
        total_vaccinations_per_hundred, people_vaccinated_per_hundred,
        people_fully_vaccinated_per_hundred, total_boosters_per_hundred,
        daily_vaccinations_per_hundred, daily_people_vaccinated,
        daily_people_vaccinated_per_hundred
    ] + list(continent_data.values())

    print(input_data)
    result = predict(input_data)
    st.write('Predicted death:', result)
