import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from data_loader import load_data
from model import train_model
from utils import predict_yield

# -----------------------------
# LOAD DATA + MODEL
# -----------------------------
data = load_data()
model, mse, r2 = train_model(data)

st.title("🌾 Climate Impact on Crop Yield")

# -----------------------------
# 🔮 PREDICT (NOW AT TOP)
# -----------------------------
st.subheader("🔮 Predict Crop Yield")

temp = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
rain = st.number_input("Rainfall (mm)", 0.0, 500.0, 100.0)

if st.button("Predict"):
    result = predict_yield(model, temp, rain)
    st.success(f"Predicted Yield: {result:.2f} tonnes/hectare")

# -----------------------------
# 📂 DATASET
# -----------------------------
st.subheader("📂 Dataset")
st.write(data)

# -----------------------------
# 📊 CORRELATION MATRIX
# -----------------------------
st.subheader("📊 Correlation Matrix")

corr = data.corr()

fig, ax = plt.subplots()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# -----------------------------
# 📈 SCATTER PLOT
# -----------------------------
st.subheader("📈 Temperature vs Yield")

fig2, ax2 = plt.subplots()
ax2.scatter(data['temperature'], data['yield'])
ax2.set_xlabel("Temperature")
ax2.set_ylabel("Yield")
st.pyplot(fig2)

# -----------------------------
# 📉 MODEL EVALUATION (NOW LAST)
# -----------------------------
st.subheader("📉 Model Evaluation")

st.write(f"MSE: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")