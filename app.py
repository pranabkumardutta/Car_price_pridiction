import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Title
st.title("🚗 Car Price Prediction App")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("car_price_prediction_.csv")
    df['Car Age'] = 2025 - df['Year']
    df.drop(columns=['Car ID', 'Year'], inplace=True)
    return df

df = load_data()

# Show dataset preview
if st.checkbox("Show Dataset"):
    st.dataframe(df)

# Define features
cat_cols = ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
num_cols = ["Car Age", "Engine Size", "Mileage"]

# Preprocessing & model
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Train model
X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# Save model
joblib.dump(model, "car_price_model.pkl")



# User input for prediction
st.header("Enter Car Details")

brand = st.selectbox("Brand", df["Brand"].unique())
fuel = st.selectbox("Fuel Type", df["Fuel Type"].unique())
transmission = st.selectbox("Transmission", df["Transmission"].unique())
condition = st.selectbox("Condition", df["Condition"].unique())
model_name = st.selectbox("Model", df["Model"].unique())
car_age = st.slider("Car Age", 0, 20, 5)
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=8.0, value=2.0, step=0.1)
mileage = st.number_input("Mileage (km)", min_value=0, max_value=300000, value=50000, step=1000)

if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Brand": [brand],
        "Fuel Type": [fuel],
        "Transmission": [transmission],
        "Condition": [condition],
        "Model": [model_name],
        "Car Age": [car_age],
        "Engine Size": [engine_size],
        "Mileage": [mileage]
    })
    

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Estimated Price: ₹{prediction:,.2f}")

    # Evaluate model
y_pred = model.predict(X_test)
st.write("### Model Performance:")
st.write(f"**Mean Squared Error:** {mean_squared_error(y_test, y_pred):.2f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")