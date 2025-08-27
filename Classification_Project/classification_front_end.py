import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

st.title("Credit Default Classification")

uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    target = st.selectbox("Select target column", df.columns)
    features = st.multiselect("Select feature columns", [col for col in df.columns if col != target])

    if features and target:
        X = df[features]
        y = df[target]

        test_size = st.slider("Test size (fraction)", 0.1, 0.5, 0.15)
        random_state = st.number_input("Random state", value=1)

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=int(random_state)
        )

        model_type = st.selectbox("Select model", ["Logistic Regression", "Bagging Classifier"])
        if model_type == "Logistic Regression":
            clf = LogisticRegression(random_state=0)
        else:
            clf = BaggingClassifier(n_estimators=50, random_state=0)

        if st.button("Train Model"):
            trained_model = clf.fit(X_train, y_train)
            y_pred = trained_model.predict(X_test)
            train_acc = accuracy_score(y_train, trained_model.predict(X_train))
            test_acc = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)

            st.write(f"Train Accuracy: {train_acc:.2f}")
            st.write(f"Test Accuracy: {test_acc:.2f}")
            st.write("Confusion Matrix:")
            st.write(cm)