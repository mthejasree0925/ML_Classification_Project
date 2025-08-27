import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("default of credit card clients.csv")  # Change to your CSV file name

# Select features and target
target = "default payment next month"  # Change to your target column
features = [col for col in df.columns if col != target and col != "ID"]  # Exclude ID if present

X = df[features]
y = df[target]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.15, random_state=1)

# Train model
clf = LogisticRegression(random_state=0)
trained_model = clf.fit(X_train, y_train)

# Predict
y_pred = trained_model.predict(X_test)

# Evaluate
train_acc = accuracy_score(y_train, trained_model.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)
print("Confusion Matrix:\n", cm)

# Save predictions
pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv("predictions.csv", index=False)