from PredictionService import generate_prediction

features = {
    "RowNumber": 1223,
    "CustomerId": 15600700,
    "Surname": "Pan",
    "CreditScore": 890,
    "Geography": "France",
    "Gender": "Male",
    "Age": 19,
    "Tenure": 1,
    "Balance": 1000,
    "NumOfProducts": 2,
    "HasCrCard": 2,
    "IsActiveMember": 2,
    "EstimatedSalary": 350
}

try:
    prediction = generate_prediction(features)
    print(f"Prediction: {prediction}")
except ValueError as e:
    print(f"Error: {e}")