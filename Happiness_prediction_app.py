import streamlit as st
import numpy as np
import joblib
import pandas as pd
import statistics as stats

# 1. Page Configuration
st.set_page_config(page_title="Happiness Predictor", page_icon="🌍", layout="centered")

# 2. Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('Happiness rank predictor.joblib')

model = load_model()

# 3. Header
st.title("🌍 Global Country Happiness Rank Predictor")
st.write("Adjust the ranking indicators below to predict a country's Happiness Rank:")

# 4. Two columns of slider inputs
col1, col2 = st.columns(2)

with col1:
    hunger = st.slider("Global Hunger Rank", 1, 217, 100)
    gdp = st.slider("GDP Per Capita Rank", 1, 217, 100)
    life_exp = st.slider("Life Expectancy Rank", 1, 217, 100)
    corruption = st.slider("Corruption Perception Rank", 1, 217, 100)
    

with col2:
    democracy = st.slider("Democracy Rank", 1, 217, 100)
    gini = st.slider("Gini_Rank", 1,217,100)
    press = st.slider("Press Freedom Rank", 1, 217, 100)
    peace = st.slider("Global Peace Rank", 1, 217, 100)
    env = st.slider("Environmental Performance Rank", 1, 217, 100)

trained_columns = [
 'Global_Hunger_Rank',
 'GDP_Per_Capita_Rank',
 'Life_Expectancy_Rank',
 'Corruption_Perception_Rank',
 'Democracy_Rank',
 'Gini_Rank',
 'Press_Freedom_Rank',
 'Global_Peace_Rank',
 'Environmental_Performance_Rank'] 
# 5. Predict Button
if st.button("Predict Happiness Rank", type="primary"):
    # Must be in the exact order the model was trained on
    #This dataframe must be the same number as the number of features in the trained model
    input_data = pd.DataFrame([{
       "Global_Hunger_Rank": hunger,
       "GDP_Per_Capita_Rank": gdp,
       "Life_Expectancy_Rank": life_exp,
       "Corruption_Perception_Rank": corruption,
       "Democracy_Rank": democracy,
       "Gini_Rank": gini,
       "Press_Freedom_Rank": press,
       "Global_Peace_Rank": peace,
       "Environmental_Performance_Rank": env
    }])
    input_data = input_data.reindex(columns=trained_columns)
    average_rank = input_data.mean(axis=1).values[0]



    predicted_rank = int(np.clip(np.round(average_rank), 1, 217))
    
    st.success(f"### 🎯 Predicted Happiness Rank: **#{predicted_rank}** (out of 217)")

