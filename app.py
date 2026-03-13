import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Used Car Dashboard", layout="wide")

st.title("🚗 Used Car Price Prediction Dashboard")

# -----------------------
# Load data
# -----------------------

data = pd.read_csv("cars.csv")

data = data.dropna()

data_encoded = pd.get_dummies(data, drop_first=True)

X = data_encoded.drop("price", axis=1)
y = data_encoded["price"]

model = RandomForestRegressor()
model.fit(X, y)

cols = X.columns


# -----------------------
# Sidebar inputs
# -----------------------

st.sidebar.header("Enter Car Details")

year = st.sidebar.number_input("Year", value=2017)
distance = st.sidebar.number_input("Distance travelled (kms)", value=5000)
car_age = st.sidebar.number_input("Car age", value=5)

brand_cols = [c for c in cols if c.startswith("brand_")]
brands = [b.replace("brand_", "") for b in brand_cols]
brand_selected = st.sidebar.selectbox("Brand", brands)

city_cols = [c for c in cols if c.startswith("city_")]
cities = [c.replace("city_", "") for c in city_cols]
city_selected = st.sidebar.selectbox("City", cities)

fuel_cols = [c for c in cols if c.startswith("fuel_type_")]
fuels = [f.replace("fuel_type_", "") for f in fuel_cols]
fuel_selected = st.sidebar.selectbox("Fuel", fuels)


# -----------------------
# Charts
# -----------------------

st.subheader("Dataset Charts")

col1, col2 = st.columns(2)

with col1:
    st.write("Price Distribution")
    st.bar_chart(data["price"])

with col2:
    st.write("Brand Count")
    st.bar_chart(data["brand"].value_counts())

st.write("City Count")
st.bar_chart(data["city"].value_counts())


# -----------------------
# Prediction
# -----------------------

if st.button("Predict"):

    df = pd.DataFrame(columns=cols)
    df.loc[0] = 0

    if "year" in df.columns:
        df.at[0, "year"] = year

    if "distance_travelled(kms)" in df.columns:
        df.at[0, "distance_travelled(kms)"] = distance

    if "car_age" in df.columns:
        df.at[0, "car_age"] = car_age

    brand_col = "brand_" + brand_selected
    if brand_col in df.columns:
        df.at[0, brand_col] = 1

    city_col = "city_" + city_selected
    if city_col in df.columns:
        df.at[0, city_col] = 1

    fuel_col = "fuel_type_" + fuel_selected
    if fuel_col in df.columns:
        df.at[0, fuel_col] = 1

    price = model.predict(df)

    st.success(f"Predicted price: ₹ {price[0]:,.0f}")