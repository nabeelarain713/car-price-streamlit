# streamlit_app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import altair as alt

# -------------------------
# Load saved model and preprocessor
# -------------------------
model = joblib.load('car_price_xgb_model.pkl')
preprocessor = joblib.load('xgb_preprocessor.pkl')

# -------------------------
# Streamlit App
# -------------------------
st.title("Car Price Prediction App 🚗💰")
st.write("Upload your test data CSV to get price predictions and visualize results.")

# File uploader
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load test data
    test_df = pd.read_csv(uploaded_file)
    
    if 'Actual Price' not in test_df.columns:
        st.error("CSV must contain 'Actual Price' column along with features.")
    else:
        # Separate features and actual price
        X_test = test_df.drop('Actual Price', axis=1)
        y_actual = test_df['Actual Price']
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Add predictions to dataframe
        results_df = X_test.copy()
        results_df['Actual Price'] = y_actual
        results_df['Predicted Price'] = y_pred
        
        st.subheader("Prediction Results")
        st.dataframe(results_df.head(20))  # Show first 20 rows

        # -------------------------
        # Bar chart visualization
        # -------------------------
        st.subheader("Actual vs Predicted Price Bar Chart")
        # Melt dataframe for charting
        chart_df = results_df[['Actual Price', 'Predicted Price']].reset_index().melt(
            id_vars='index', value_vars=['Actual Price', 'Predicted Price'],
            var_name='Price Type', value_name='Price'
        )

        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('index:N', title='Car Index'),
            y=alt.Y('Price:Q'),
            color='Price Type:N',
            tooltip=['index', 'Price Type', 'Price']
        ).properties(width=800, height=400)

        st.altair_chart(chart, use_container_width=True)

        # -------------------------
        # Metrics
        # -------------------------
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        st.subheader("Evaluation Metrics")
        st.write(f"**MAE:** ${mae:.2f}")
        st.write(f"**RMSE:** ${rmse:.2f}")
        st.write(f"**R² Score:** {r2:.4f}")