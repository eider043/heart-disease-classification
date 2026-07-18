import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = {
    "age": "Edad",
    "sex": "Sexo",
    "cp": "Tipo dolor pecho",
    "trestbps": "Presion arterial",
    "chol": "Colesterol",
    "fbs": "Glucosa ayunas",
    "restecg": "ECG reposo",
    "thalach": "Frec. cardiaca max",
    "exang": "Angina por ejercicio",
    "oldpeak": "Depresion ST",
    "slope": "Pendiente ST",
    "ca": "Vasos coloreados",
    "thal": "Talasemia"
}

def preprocess(df, target="target", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

    print(f"Train: {X_train_scaled.shape} | Test: {X_test_scaled.shape}")
    print(f"Prevalencia train: {y_train.mean():.2%} | test: {y_test.mean():.2%}")
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), scaler