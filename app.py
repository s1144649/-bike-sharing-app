import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import sqlite3
from datetime import datetime

# ---------------------------
# DATABASE SETUP
# ---------------------------
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

# ---------------------------
# CACHED FUNCTIONS
# ---------------------------
@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

@st.cache_data
def load_predictions():
    conn = sqlite3.connect('predictions.db')
    df = pd.read_sql_query(
        'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT 100',
        conn
    )
    conn.close()
    return df

model = load_model()

# ---------------------------
# UI
# ---------------------------
st.title('🚲 Bike Sharing Voorspeller')
st.markdown('Vul de onderstaande kenmerken in om het aantal fietsverhuren te voorspellen.')

# Tijd
st.header('⏰ Tijd')
hr = st.slider('Uur van de dag', 0, 23, 8)
yr = st.selectbox('Jaar', [0, 1], format_func=lambda x: '2011' if x == 0 else '2012')

# Kalender
st.header('📅 Kalender')
season = st.selectbox('Seizoen', [1, 2, 3, 4],
    format_func=lambda x: {1: 'Lente', 2: 'Zomer', 3: 'Herfst', 4: 'Winter'}[x])

weekday = st.selectbox('Dag van de week', list(range(7)),
    format_func=lambda x: ['Zondag','Maandag','Dinsdag','Woensdag','Donderdag','Vrijdag','Zaterdag'][x])

holiday = st.selectbox('Feestdag', [0, 1],
    format_func=lambda x: 'Nee' if x == 0 else 'Ja')

workingday = st.selectbox('Werkdag', [0, 1],
    format_func=lambda x: 'Nee' if x == 0 else 'Ja')

# Weer
st.header('🌤️ Weer')
weathersit = st.selectbox('Weersomstandigheden', [1, 2, 3, 4],
    format_func=lambda x: {
        1: 'Helder',
        2: 'Mistig',
        3: 'Lichte regen/sneeuw',
        4: 'Zware regen/sneeuw'
    }[x])

temp_c = st.slider('Temperatuur (°C)', -8, 39, 20)
hum_pct = st.slider('Luchtvochtigheid (%)', 0, 100, 50)
wind_kmh = st.slider('Windsnelheid (km/h)', 0, 67, 15)

# ---------------------------
# PREPROCESSING
# ---------------------------
temp = (temp_c + 8) / 47
hum = hum_pct / 100
windspeed = wind_kmh / 67

input_data = np.array([[
    temp, hum, windspeed, hr, season,
    weathersit, weekday, holiday, workingday, yr
]])

# ---------------------------
# PREDICTION
# ---------------------------
if st.button('Voorspel aantal verhuren 🚲'):
    prediction = model.predict(input_data)
    pred_value = int(prediction[0])

    # Save ONLY when button is pressed
    save_prediction(
        hr, yr, season, weekday, holiday,
        workingday, weathersit,
        temp_c, hum_pct, wind_kmh,
        pred_value
    )

    st.success(f'Voorspeld aantal fietsverhuren: **{pred_value}**')

# ---------------------------
# DISPLAY STORED DATA
# ---------------------------
st.divider()
st.header('📋 Opgeslagen Voorspellingen')

df_preds = load_predictions()

if df_preds.empty:
    st.info('Nog geen voorspellingen opgeslagen.')
else:
    st.write(f'Totaal aantal (laatste 100): **{len(df_preds)}**')
    st.dataframe(df_preds, use_container_width=True)

    # Plot
    st.subheader('📈 Voorspellingen over tijd')
    df_preds['timestamp'] = pd.to_datetime(df_preds['timestamp'])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df_preds['timestamp'], df_preds['prediction'], marker='o', linewidth=1)
    ax.set_xlabel('Tijdstip')
    ax.set_ylabel('Voorspeld aantal verhuren')
    ax.set_title('Voorspellingen over tijd')

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
    plt.close(fig)  # prevents memory leak