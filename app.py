import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.joblib')

st.title('🚲 Bike Sharing Voorspeller')
st.markdown('Vul de onderstaande kenmerken in om het aantal fietsverhuren te voorspellen.')

st.header('⏰ Tijd')
hr = st.slider('Uur van de dag', 0, 23, 8)
yr = st.selectbox('Jaar', options=[0, 1], format_func=lambda x: '2011' if x == 0 else '2012')

st.header('📅 Kalender')
season = st.selectbox('Seizoen', options=[1, 2, 3, 4],
    format_func=lambda x: {1: 'Lente', 2: 'Zomer', 3: 'Herfst', 4: 'Winter'}[x])
weekday = st.selectbox('Dag van de week', options=list(range(7)),
    format_func=lambda x: ['Zondag','Maandag','Dinsdag','Woensdag','Donderdag','Vrijdag','Zaterdag'][x])
holiday = st.selectbox('Feestdag', options=[0, 1],
    format_func=lambda x: 'Nee' if x == 0 else 'Ja')
workingday = st.selectbox('Werkdag', options=[0, 1],
    format_func=lambda x: 'Nee' if x == 0 else 'Ja')

st.header('🌤️ Weer')
weathersit = st.selectbox('Weersomstandigheden', options=[1, 2, 3, 4],
    format_func=lambda x: {
        1: 'Helder',
        2: 'Mistig',
        3: 'Lichte regen/sneeuw',
        4: 'Zware regen/sneeuw'
    }[x])
temp_c = st.slider('Temperatuur (°C)', -8, 39, 20)
hum_pct = st.slider('Luchtvochtigheid (%)', 0, 100, 50)
wind_kmh = st.slider('Windsnelheid (km/h)', 0, 67, 15)

# Normaliseer terug naar modelwaarden
temp = (temp_c + 8) / 47        # schaal: -8°C tot 39°C
hum = hum_pct / 100             # schaal: 0% tot 100%
windspeed = wind_kmh / 67       # schaal: 0 tot 67 km/h

# Predict
input_data = np.array([[temp, hum, windspeed, hr, season, weathersit, weekday, holiday, workingday, yr]])

if st.button('Voorspel aantal verhuren 🚲'):
    prediction = model.predict(input_data)
    st.success(f'Voorspeld aantal fietsverhuren: **{int(prediction[0])}**')
