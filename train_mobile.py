import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
data = pd.read_csv("datasets/mobile.csv")

# Convert Brand names to numbers
encoder = LabelEncoder()
data["Brand"] = encoder.fit_transform(data["Brand"])

# Features and target
X = data.drop("Price", axis=1)
y = data["Price"]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

# Save model and encoder
with open("models/mobile_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/mobile_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("Mobile model created successfully!")