import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.write("""
# Insurance Charges Prediction App
This app predicts the **insurance charges** based on user inputs using a Random Forest model!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    age = st.sidebar.slider('Age', 18, 100, 30)
    sex = st.sidebar.selectbox('Sex', ('male', 'female'))
    bmi = st.sidebar.slider('BMI', 10.0, 50.0, 25.0)
    children = st.sidebar.slider('Children', 0, 10, 0)
    smoker = st.sidebar.selectbox('Smoker', ('yes', 'no'))
    region = st.sidebar.selectbox('Region', ('northeast', 'northwest', 'southeast', 'southwest'))

    # Encode categorical features for model input
    sex_encoded = 1 if sex == 'male' else 0
    smoker_encoded = 1 if smoker == 'yes' else 0

    data = {
        'age': age,
        'sex_male': sex_encoded,
        'bmi': bmi,
        'children': children,
        'smoker_yes': smoker_encoded,
        'region_northwest': 1 if region == 'northwest' else 0,
        'region_southeast': 1 if region == 'southeast' else 0,
        'region_southwest': 1 if region == 'southwest' else 0
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Load dataset and preprocess
data = pd.read_csv('insurance.csv')
data = pd.get_dummies(data, columns=['sex', 'region', 'smoker'], drop_first=True)
X = data.drop('charges', axis=1)
y = data['charges']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = user_input_features()

# Ensure user input features match the training data
df = pd.get_dummies(df)
missing_cols = set(X_train.columns) - set(df.columns)
for col in missing_cols:
    df[col] = 0
df = df[X_train.columns]

st.subheader('User Input parameters')
st.write(df)

# Train the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict using user input
prediction = model.predict(df)

# Evaluate the model on test data
model_pred = model.predict(X_test)
model_MSE = np.sqrt(mean_squared_error(y_test, model_pred))
model_R2 = r2_score(y_test, model_pred)

st.subheader('Model Evaluation')
st.write(f"Root Mean Squared Error (RMSE): {model_MSE:.2f}")
st.write(f"R-squared (R2): {model_R2:.2f}")

st.subheader('Prediction')
st.write(f"Predicted Insurance Charges: ${prediction[0]:,.2f}")
