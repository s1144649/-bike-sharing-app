import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import sqlite3
from datetime import datetime


def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            hr INTEGER,
            yr INTEGER,
            season INTEGER,
            weekday INTEGER,
            holiday INTEGER,
            workingday INTEGER,
            weathersit INTEGER,
            temp_c REAL,
            hum_pct REAL,
            wind_kmh REAL,
            prediction INTEGER
        )
    ''')
    conn.commit()
    conn.close()

init_db()

def save_prediction(hr, yr, season, weekday, holiday, workingday,
                    weathersit, temp_c, hum_pct, wind_kmh, prediction):
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions 
        (timestamp, hr, yr, season, weekday, holiday, workingday, 
         weathersit, temp_c, hum_pct, wind_kmh, prediction)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        hr, yr, season, weekday, holiday, workingday,
        weathersit, temp_c, hum_pct, wind_kmh, prediction
    ))
    conn.commit()
    conn.close()

def load_predictions():
    conn = sqlite3.connect('predictions.db')
    df_preds = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY timestamp DESC',
        conn
    )
    conn.close()
    return df_preds

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
    pred_value = int(prediction[0])

st.divider()
st.header('📋 Opgeslagen Voorspellingen')

df_preds = load_predictions()

if df_preds.empty:
    st.info('Nog geen voorspellingen opgeslagen.')
else:
    st.write(f'Totaal aantal voorspellingen: **{len(df_preds)}**')
    st.dataframe(df_preds, use_container_width=True)

if not df_preds.empty:
    st.subheader('📈 Voorspellingen over tijd')
    df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_preds['timestamp'], df_preds['prediction'],
            marker='o', color='steelblue', linewidth=1)
    ax.set_xlabel('Tijdstip')
    ax.set_ylabel('Voorspeld aantal verhuren')
    ax.set_title('Voorspellingen over tijd')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)



    # Sla op in database
    save_prediction(hr, yr, season, weekday, holiday, workingday,
                    weathersit, temp_c, hum_pct, wind_kmh, pred_value)

    st.success(f'Voorspeld aantal fietsverhuren: **{pred_value}**')
