import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import joblib

data_path = r"C:\Users\ilyes\OneDrive\Documents\GitHub\ASL2\data collection\data.csv"

data = pd.read_csv(data_path)
inputs = [
    "x1", "y1", "z1", "x2", "y2", "z2", "x3", "y3", "z3", "x4", "y4", "z4",
    "x5", "y5", "z5", "x6", "y6", "z6", "x7", "y7", "z7", "x8", "y8", "z8",
    "x9", "y9", "z9", "x10", "y10", "z10", "x11", "y11", "z11", "x12", "y12", "z12",
    "x13", "y13", "z13", "x14", "y14", "z14", "x15", "y15", "z15", "x16", "y16", "z16",
    "x17", "y17", "z17", "x18", "y18", "z18", "x19", "y19", "z19", "x20", "y20", "z20",
    "x21", "y21", "z21", "f1", "f2", "f3", "f4", "f5"
]

# Convert string labels to numerical values
label_encoder = LabelEncoder()
data['alpha_numeric'] = label_encoder.fit_transform(data['alpha'])

y = data['alpha_numeric']
X = data[inputs]

print("Unique classes:", label_encoder.classes_)
print("Numerical mapping:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1, test_size=0.2)

# Define XGBoost model
xgb_model = XGBClassifier(random_state=1, eval_metric='mlogloss')
xgb_model.fit(train_X, train_y)
xgb_val_predictions = xgb_model.predict(val_X)

# Convert predictions back to original labels for interpretation
val_predictions_original = label_encoder.inverse_transform(xgb_val_predictions)
val_y_original = label_encoder.inverse_transform(val_y)

print("\nValidation True Labels:", val_y_original[:10])
print("Validation Predictions:", val_predictions_original[:10])
print("Accuracy:", accuracy_score(val_y, xgb_val_predictions))

# Save both the model and the label encoder
joblib.dump(xgb_model, "model_asl.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")

print("\nModel and label encoder saved successfully!")