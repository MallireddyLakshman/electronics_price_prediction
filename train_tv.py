import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

data = pd.read_csv("datasets/tv.csv")

encoder = LabelEncoder()
data["Brand"] = encoder.fit_transform(data["Brand"])

X = data.drop("Price", axis=1)
y = data["Price"]

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

with open("models/tv_model.pkl", "wb") as file:
    pickle.dump(model, file)

with open("models/tv_encoder.pkl", "wb") as file:
    pickle.dump(encoder, file)

print("TV model created successfully!")