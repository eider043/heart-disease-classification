from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import joblib
import os

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
}

def train_all(X_train, y_train, cv=5):
    os.makedirs("../models", exist_ok=True)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    results = {}

    for name, model in MODELS.items():
        scores = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring="roc_auc")
        model.fit(X_train, y_train)
        joblib.dump(model, f"../models/{name.replace(' ', '_').lower()}.pkl")

        results[name] = {
            "model": model,
            "cv_auc_mean": scores.mean(),
            "cv_auc_std": scores.std()
        }
        print(f"{name:25s} | CV AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results