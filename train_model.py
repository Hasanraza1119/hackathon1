import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib
import os

# Load dataset
df = pd.read_csv("healthcare_dataset.csv")
df.dropna(inplace=True)

# Encode target labels
target_column = "Test Results"
le = LabelEncoder()
df[target_column] = le.fit_transform(df[target_column])

# Split features and target
X = df.drop(columns=[target_column])
y = df[target_column]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Create models directory if not exists
os.makedirs("models", exist_ok=True)

# Save model and encoder
joblib.dump(model, "models/trained_model.pkl")
joblib.dump(le, "models/label_encoder.pkl")

# Evaluation printout
print("✅ Model training complete!")
print("📁 Saved: models/trained_model.pkl")
print("📁 Saved: models/label_encoder.pkl")
print("🧪 Test Accuracy:")
print(classification_report(y_test, model.predict(X_test)))
