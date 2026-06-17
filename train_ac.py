import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("datasets/ac.csv")

encoder = LabelEncoder()
data["Brand"] = encoder.fit_transform(data["Brand"])

X = data.drop("Price", axis=1)
y = data["Price"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

with open("models/ac_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/ac_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("AC model created successfully!")