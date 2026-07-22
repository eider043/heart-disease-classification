"""
Dashboard interactivo — Deteccion de Enfermedades Cardiacas
Autor: Eider
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Cargar datos y modelos ──────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, "data", "heart_disease.csv"))

MODEL_NAMES = [
    "logistic_regression", "svm", "decision_tree",
    "random_forest", "gradient_boosting", "knn"
]
DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "svm": "SVM",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "knn": "KNN"
}

models = {}
for m in MODEL_NAMES:
    try:
        models[DISPLAY_NAMES[m]] = joblib.load(
            os.path.join(BASE_DIR, "models", f"{m}.pkl")
        )
    except FileNotFoundError:
        pass

metrics_df = pd.read_csv(os.path.join(BASE_DIR, "outputs", "model_metrics.csv"))

scaler = StandardScaler()
X = df.drop(columns=["target"])
y = df["target"]
scaler.fit(X)
X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)

# ── App ──────────────────────────────────────────────────────────────
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "Heart Disease Classification Dashboard"

CARD_STYLE = {"border": "1px solid #444", "borderRadius": "8px", "padding": "15px",
              "backgroundColor": "#2d2d2d", "marginBottom": "15px"}

total = len(df)
positivos = df["target"].sum()
prevalencia = positivos / total
best_auc = metrics_df["AUC"].max()
best_model_name = metrics_df.loc[metrics_df["AUC"].idxmax(), "Model"]

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Heart Disease Classification Dashboard",
                    style={"color": "#e74c3c", "fontWeight": "bold", "marginTop": "20px"}),
            html.P("Analisis comparativo de modelos de clasificacion sobre el dataset Heart Disease UCI",
                   style={"color": "#aaa"})
        ])
    ]),

    html.Hr(style={"borderColor": "#444"}),

    dbc.Row([
        dbc.Col(dbc.Card([
            html.H4(f"{total}", style={"color": "#3498db", "fontWeight": "bold", "fontSize": "2rem"}),
            html.P("Total pacientes", style={"color": "#aaa", "margin": 0})
        ], body=True, style=CARD_STYLE), width=3),
        dbc.Col(dbc.Card([
            html.H4(f"{positivos}", style={"color": "#e74c3c", "fontWeight": "bold", "fontSize": "2rem"}),
            html.P("Con enfermedad cardiaca", style={"color": "#aaa", "margin": 0})
        ], body=True, style=CARD_STYLE), width=3),
        dbc.Col(dbc.Card([
            html.H4(f"{prevalencia:.1%}", style={"color": "#f39c12", "fontWeight": "bold", "fontSize": "2rem"}),
            html.P("Prevalencia en muestra", style={"color": "#aaa", "margin": 0})
        ], body=True, style=CARD_STYLE), width=3),
        dbc.Col(dbc.Card([
            html.H4(f"{best_auc:.4f}", style={"color": "#2ecc71", "fontWeight": "bold", "fontSize": "2rem"}),
            html.P(f"Mejor AUC ({best_model_name})", style={"color": "#aaa", "margin": 0})
        ], body=True, style=CARD_STYLE), width=3),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Comparacion de Modelos", style={"color": "#fff"}),
            dcc.Graph(id="bar-metrics")
        ], width=6),
        dbc.Col([
            html.H5("Curvas ROC", style={"color": "#fff"}),
            dcc.Graph(id="roc-curves")
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Metrica de comparacion:", style={"color": "#aaa"}),
            dcc.RadioItems(
                id="metric-selector",
                options=[
                    {"label": " AUC-ROC", "value": "AUC"},
                    {"label": " Accuracy", "value": "Accuracy"},
                    {"label": " F1 Score", "value": "F1"},
                ],
                value="AUC",
                inline=True,
                style={"color": "#fff"}
            )
        ], width=6),
        dbc.Col([
            html.Label("Modelo para matriz de confusion:", style={"color": "#aaa"}),
            dcc.Dropdown(
                id="model-selector",
                options=[{"label": m, "value": m} for m in models.keys()],
                value=best_model_name,
                style={"color": "#000"}
            )
        ], width=6),
    ], style={"marginBottom": "15px"}),

    dbc.Row([
        dbc.Col([
            html.H5("Matriz de Confusion", style={"color": "#fff"}),
            dcc.Graph(id="confusion-matrix")
        ], width=6),
        dbc.Col([
            html.H5("Distribucion por Variable Clinica", style={"color": "#fff"}),
            dcc.Dropdown(
                id="feature-selector",
                options=[{"label": c, "value": c} for c in X.columns],
                value="age",
                style={"color": "#000", "marginBottom": "10px"}
            ),
            dcc.Graph(id="feature-dist")
        ], width=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Importancia de Variables — Random Forest", style={"color": "#fff"}),
            dcc.Graph(id="feature-importance")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.H5("Tabla de Resultados", style={"color": "#fff"}),
            dash_table.DataTable(
                data=metrics_df.round(4).to_dict("records"),
                columns=[{"name": c, "id": c} for c in metrics_df.columns],
                style_header={"backgroundColor": "#e74c3c", "color": "white", "fontWeight": "bold"},
                style_data={"backgroundColor": "#2d2d2d", "color": "white"},
                style_data_conditional=[
                    {"if": {"row_index": 0},
                     "backgroundColor": "#1a3a1a", "fontWeight": "bold"}
                ],
                sort_action="native"
            )
        ])
    ], style={"marginTop": "10px", "marginBottom": "30px"})

], fluid=True, style={"backgroundColor": "#1a1a2e", "minHeight": "100vh"})


# ── Callbacks ────────────────────────────────────────────────────────

@app.callback(Output("bar-metrics", "figure"), Input("metric-selector", "value"))
def update_bar(metric):
    df_sorted = metrics_df.sort_values(metric, ascending=True)
    colors = ["#e74c3c" if v == df_sorted[metric].max() else "#3498db"
              for v in df_sorted[metric]]
    fig = go.Figure(go.Bar(
        x=df_sorted[metric], y=df_sorted["Model"],
        orientation="h", marker_color=colors,
        text=[f"{v:.4f}" for v in df_sorted[metric]],
        textposition="outside"
    ))
    fig.update_layout(
        template="plotly_dark", xaxis_range=[0, 1.1],
        xaxis_title=metric, margin=dict(l=10, r=30, t=20, b=20),
        plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d"
    )
    return fig


@app.callback(Output("roc-curves", "figure"), Input("metric-selector", "value"))
def update_roc(_):
    fig = go.Figure()
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc = roc_auc_score(y, y_prob)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  name=f"{name} (AUC={auc:.3f})",
                                  line=dict(color=color, width=2)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              name="Aleatorio", line=dict(dash="dash", color="gray")))
    fig.update_layout(
        template="plotly_dark", xaxis_title="FPR", yaxis_title="TPR",
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d",
        legend=dict(font=dict(size=10))
    )
    return fig


@app.callback(Output("confusion-matrix", "figure"), Input("model-selector", "value"))
def update_cm(model_name):
    if model_name not in models:
        return go.Figure()
    y_pred = models[model_name].predict(X_scaled)
    cm = confusion_matrix(y, y_pred)
    labels = ["Sin enfermedad", "Con enfermedad"]
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=labels, y=labels, aspect="auto")
    fig.update_layout(
        template="plotly_dark", xaxis_title="Predicho", yaxis_title="Real",
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d"
    )
    return fig


@app.callback(Output("feature-dist", "figure"), Input("feature-selector", "value"))
def update_dist(feature):
    fig = px.histogram(df, x=feature, color="target", barmode="overlay",
                       color_discrete_map={0: "#3498db", 1: "#e74c3c"},
                       labels={"target": "Enfermedad", feature: feature},
                       opacity=0.75)
    fig.update_layout(
        template="plotly_dark", margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d"
    )
    return fig


@app.callback(Output("feature-importance", "figure"), Input("model-selector", "value"))
def update_importance(_):
    if "Random Forest" not in models:
        return go.Figure()
    model = models["Random Forest"]
    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    colors = ["#e74c3c" if v >= importances.quantile(0.75) else "#3498db"
              for v in importances]
    fig = go.Figure(go.Bar(
        x=importances.values, y=importances.index,
        orientation="h", marker_color=colors
    ))
    fig.update_layout(
        template="plotly_dark", xaxis_title="Importancia relativa",
        margin=dict(l=10, r=10, t=20, b=20),
        plot_bgcolor="#2d2d2d", paper_bgcolor="#2d2d2d"
    )
    return fig


if __name__ == "__main__":
    app.run(debug=True)