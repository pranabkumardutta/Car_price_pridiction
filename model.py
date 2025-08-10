import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("car_price_prediction_.csv")
df

df.info()

df['Car Age'] = 2025 - df['Year']
df.drop(columns=['Car ID','Year'], inplace=True)

df
X = df.drop(columns=["Price"])
y = df["Price"]

cat_cols= ["Brand", "Fuel Type", "Transmission", "Condition", "Model"]
num_cols = ["Car Age", "Engine Size", "Mileage"]

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
    ('num', StandardScaler(), num_cols)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

import joblib
joblib.dump(model, "car_price_model.pkl")
print("✅ Model saved as car_price_model.pkl")