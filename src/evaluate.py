import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score,
                              confusion_matrix, roc_curve, classification_report)
import os

os.makedirs("../outputs", exist_ok=True)

def evaluate_all(trained_models, X_test, y_test):
    metrics = []
    for name, info in trained_models.items():
        model = info["model"]
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        metrics.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_prob),
            "CV_AUC": info["cv_auc_mean"]
        })

    df_metrics = pd.DataFrame(metrics).sort_values("AUC", ascending=False)
    df_metrics.to_csv("../outputs/model_metrics.csv", index=False)
    print("\n--- Resultados en Test Set ---")
    print(df_metrics.to_string(index=False))
    return df_metrics

def plot_results(trained_models, X_test, y_test, df_metrics):
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 3, figure=fig)

    # 1. Comparacion AUC
    ax1 = fig.add_subplot(gs[0, :2])
    colors = ["#e74c3c" if r == df_metrics["AUC"].max() else "#3498db" for r in df_metrics["AUC"]]
    bars = ax1.barh(df_metrics["Model"], df_metrics["AUC"], color=colors, edgecolor="white")
    ax1.set_title("Comparacion AUC-ROC por Modelo", fontsize=13, fontweight="bold")
    ax1.set_xlabel("AUC-ROC")
    ax1.set_xlim(0, 1.1)
    ax1.axvline(0.9, color="green", linestyle="--", alpha=0.5, label="Umbral excelente")
    for bar, val in zip(bars, df_metrics["AUC"]):
        ax1.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontweight="bold")
    ax1.legend()
    ax1.grid(axis="x", alpha=0.3)

    # 2. Accuracy vs F1
    ax2 = fig.add_subplot(gs[0, 2])
    x = range(len(df_metrics))
    ax2.bar([i - 0.2 for i in x], df_metrics["Accuracy"], width=0.4,
            label="Accuracy", color="#2ecc71", alpha=0.85)
    ax2.bar([i + 0.2 for i in x], df_metrics["F1"], width=0.4,
            label="F1 Score", color="#f39c12", alpha=0.85)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([m.split()[0] for m in df_metrics["Model"]], rotation=45)
    ax2.set_title("Accuracy vs F1 Score", fontsize=13, fontweight="bold")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)

    # 3. Curvas ROC
    ax3 = fig.add_subplot(gs[1, :2])
    colors_roc = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for (name, info), color in zip(trained_models.items(), colors_roc):
        y_prob = info["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax3.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC={auc:.3f})")
    ax3.plot([0, 1], [0, 1], "k--", lw=1.5, label="Aleatorio")
    ax3.set_title("Curvas ROC — Todos los Modelos", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Tasa Falsos Positivos")
    ax3.set_ylabel("Tasa Verdaderos Positivos")
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(alpha=0.3)

    # 4. Matriz de confusion - mejor modelo
    best_name = df_metrics.iloc[0]["Model"]
    best_model = trained_models[best_name]["model"]
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)

    ax4 = fig.add_subplot(gs[1, 2])
    im = ax4.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax4)
    ax4.set_title(f"Matriz de Confusion\n{best_name}", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Predicho")
    ax4.set_ylabel("Real")
    ax4.set_xticks([0, 1])
    ax4.set_yticks([0, 1])
    ax4.set_xticklabels(["Sin enfermedad", "Con enfermedad"])
    ax4.set_yticklabels(["Sin enfermedad", "Con enfermedad"])
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, fontweight="bold",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

    plt.suptitle("Analisis Comparativo — Clasificacion de Enfermedad Cardiaca",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig("../outputs/model_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nReporte del mejor modelo ({best_name}):")
    print(classification_report(y_test, y_pred_best,
                                 target_names=["Sin enfermedad", "Con enfermedad"]))

def plot_feature_importance(trained_models, feature_names):
    for name in ["Random Forest", "Gradient Boosting"]:
        if name in trained_models:
            model = trained_models[name]["model"]
            importances = pd.Series(model.feature_importances_, index=feature_names)
            importances = importances.sort_values(ascending=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ["#e74c3c" if v >= importances.quantile(0.75) else "#3498db"
                      for v in importances]
            importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
            ax.set_title(f"Importancia de Variables Clinicas — {name}",
                         fontsize=13, fontweight="bold")
            ax.set_xlabel("Importancia relativa")
            ax.axvline(importances.mean(), color="black", linestyle="--",
                       alpha=0.5, label="Media")
            ax.legend()
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"../outputs/feature_importance.png", dpi=150, bbox_inches="tight")
            plt.show()
            break