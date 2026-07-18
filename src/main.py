"""
Deteccion de Enfermedades Cardiacas
Clasificacion con modelos estadisticos y machine learning

Autor: Eider
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_data
from preprocessing import preprocess
from models import train_all
from evaluate import evaluate_all, plot_results, plot_feature_importance

def main():
    print("=" * 60)
    print("DETECCION DE ENFERMEDADES CARDIACAS")
    print("Clasificacion Estadistica y Machine Learning")
    print("=" * 60)

    print("\n[1/4] Cargando datos...")
    df = load_data()

    print("\n[2/4] Preprocesando...")
    X_train, X_test, y_train, y_test, scaler = preprocess(df)

    print("\n[3/4] Entrenando modelos con validacion cruzada estratificada...")
    trained_models = train_all(X_train, y_train)

    print("\n[4/4] Evaluando y generando graficas...")
    df_metrics = evaluate_all(trained_models, X_test, y_test)
    plot_results(trained_models, X_test, y_test, df_metrics)
    plot_feature_importance(trained_models, X_train.columns)

    print("\nProceso completado. Revisa outputs/ y models/")
    print("Para iniciar el dashboard ejecuta: python dashboard.py")

if __name__ == "__main__":
    main()