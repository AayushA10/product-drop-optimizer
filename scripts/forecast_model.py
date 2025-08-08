import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
df = pd.read_csv("data/sample_sales_data.csv")

# Drop unused columns
df = df.drop(columns=["ProductID", "ProductName", "LaunchDate"])

# One-hot encode categorical columns
cat_cols = ["Category", "Channel", "Region"]
encoder = OneHotEncoder(drop='first', sparse_output=False)

encoded = encoder.fit_transform(df[cat_cols])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))

# Combine with numerical columns
X = pd.concat([df[["Price"]], encoded_df], axis=1)
y = df["UnitsSold30"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"\n📉 Mean Squared Error on Test Set: {mse:.2f}")

# Show feature importance
coeff_df = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
}).sort_values(by="Coefficient", ascending=False)

print("\n🔍 Feature Importance:")
print(coeff_df)

import joblib

# Save model and encoder
joblib.dump(model, "app/model.pkl")
joblib.dump(encoder, "app/encoder.pkl")

