import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# 1. Dataset Load
df = pd.read_csv("insurance.csv")

df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

df = pd.get_dummies(df, drop_first=True)

# 2. Convert Text Data into Numbers (Encoding)
df["sex"] = df["sex"].map({"female": 0, "male": 1})
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df["region"] = df["region"].map(
    {"southwest": 0, "southeast": 1, "northwest": 2, "northeast": 3}
)

# 3. Inputs (X) and Target Output (y)
X = df[["age", "sex", "bmi", "children", "smoker", "region"]]
y = df["charges"]

# 4. Model Train
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the model in a .pkl file.
joblib.dump(model, "insurance_model.pkl")
print("✅ ML model trained successfully and saved as 'insurance_model.pkl'!")