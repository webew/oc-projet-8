import streamlit as st
import numpy as np
import pandas as pd
import requests
import plotly.graph_objects as go
import os

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
# st.write(f"DEBUG - BASE_URL: {BASE_URL}")  # à supprimer après

# ── fonctions ────────────────────────────────────────────────────
def fetch_api():
    client_id = st.session_state["selectbox_client_id"]
    
    if client_id is None:  # selectbox vidé
        st.session_state.pop("prediction", None)  # on efface la prédiction
        return
    
    r = requests.get(f"{BASE_URL}/predict/{client_id}", timeout=30)
    st.session_state["prediction"] = r.json()

def get_global_importance(feature, shap_dict):
    total = 0
    for key, val in shap_dict.items():
        clean_key = key.replace("num__", "").replace("cat__", "")
        if clean_key == feature or clean_key.startswith(feature + "_"):
            total += abs(val)
    return total

def display_gauge(probability, threshold):
    score = probability * 100
    threshold_score = threshold * 100
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"suffix": "%", "valueformat": ".1f"},
        title={"text": "Probabilité de défaut"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "rgba(0,0,0,0)"},
            "steps": [
                {"range": [0, 7], "color": "green"},
                {"range": [7, 9], "color": "orange"},
                {"range": [9, 100], "color": "red"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "thickness": 0.75, "value": threshold_score}
        }
    ))
    cx, cy = 0.5, 0.0
    gauge_width, gauge_height = 240, 120
    angle_rad = np.radians(180 - (score / 100) * 180)
    L_px = 110
    x_tip = (gauge_width + L_px * np.cos(angle_rad)) * cx / gauge_width
    y_tip = L_px * np.sin(angle_rad) / gauge_height
    fig.add_shape(type="line", x0=cx, y0=cy, x1=x_tip, y1=y_tip,
                  line={"color": "lightgray", "width": 3}, xref="paper", yref="paper")
    fig.add_shape(type="circle", x0=cx-0.02, y0=cy-0.02, x1=cx+0.02, y1=cy+0.02,
                  fillcolor="blue", line={"color": "blue"}, xref="paper", yref="paper")
    fig.update_layout(width=500, height=300)
    st.plotly_chart(fig, use_container_width=False)

def display_shap(shap_local, shap_global, n_features=10):
    top_features = sorted(shap_local, key=lambda f: abs(shap_local[f]), reverse=True)[:n_features]
    top_features = top_features[::-1]
    labels = [f.replace("num__", "").replace("cat__", "") for f in top_features]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Global", x=[shap_global.get(f, 0) for f in top_features], y=labels, orientation="h", marker_color="steelblue"))
    fig.add_trace(go.Bar(name="Ce client", x=[abs(shap_local.get(f, 0)) for f in top_features], y=labels, orientation="h", marker_color="orange"))
    fig.update_layout(
        legend={"traceorder": "reversed"}  # inverse l'affichage de la légende
    )
    fig.update_layout(title="Importance des caractéristiques : ce client vs tous les clients", barmode="group", height=400, xaxis_title="Valeur SHAP")
    st.plotly_chart(fig, use_container_width=True)

def display_distribution(client_id, feature, X_scoring):
    client_value = X_scoring.loc[client_id, feature]
    all_values = X_scoring[feature].dropna()
    fig = go.Figure()
    if pd.api.types.is_numeric_dtype(all_values):
        fig.add_trace(go.Histogram(x=all_values, name="Tous les clients", nbinsx=50, opacity=0.7))
        fig.add_vline(x=float(client_value), line_color="orange", line_width=2, annotation_text="Ce client", annotation_position="top")
    else:
        counts = all_values.value_counts().sort_index()
        labels = {0: "Non", 1: "Oui"}
        colors = ["red" if str(val) == str(client_value) else "steelblue" for val in counts.index]
        fig.add_trace(go.Bar(x=[labels.get(v, str(v)) for v in counts.index], y=counts.values, marker_color=colors))
        try:
            client_label = labels.get(int(float(client_value)), str(client_value))
        except (ValueError, TypeError):
            client_label = str(client_value)
        fig.update_layout(title=f"{feature} — Ce client : {client_label}")
    st.plotly_chart(fig, use_container_width=True)

def display_bivariate(client_id, feature_x, feature_y, X_scoring):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_scoring[feature_x], y=X_scoring[feature_y], mode="markers",
                             marker={"size": 3, "color": "steelblue", "opacity": 0.3}, name="Tous les clients"))
    fig.add_trace(go.Scatter(x=[X_scoring.loc[client_id, feature_x]], y=[X_scoring.loc[client_id, feature_y]],
                             mode="markers", marker={"size": 12, "color": "orange", "symbol": "star"}, name="Ce client"))
    fig.update_layout(title=f"{feature_x} vs {feature_y}", xaxis_title=feature_x, yaxis_title=feature_y, height=400)
    st.plotly_chart(fig, use_container_width=True)

# ── chargement des données ────────────────────────────────────────────────────
df = pd.read_csv("data/raw/application_test.csv")
X_scoring = pd.read_csv("data/processed/X_scoring.csv", index_col="SK_ID_CURR")

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Client")
    st.selectbox(
        key="selectbox_client_id",
        label="Sélectionner un client :",
        options=df['SK_ID_CURR'],
        on_change=fetch_api,
        index=None,
        placeholder="Sélectionner un client"
    )

    if "prediction" in st.session_state:
        client_id = st.session_state["selectbox_client_id"]
        prediction = st.session_state["prediction"]
        features_sorted = sorted(
            X_scoring.columns.tolist(),
            key=lambda f: get_global_importance(f, prediction["shap_local"]),
            reverse=True
        )

        # Features modifiables
        st.header("Modifier les informations")
        modified_values = {}
        for feature in features_sorted[:10]:
            current_value = X_scoring.loc[client_id, feature]
            if pd.api.types.is_numeric_dtype(X_scoring[feature]):
                modified_values[feature] = st.slider(
                    label=feature,
                    min_value=float(X_scoring[feature].min()),
                    max_value=float(X_scoring[feature].max()),
                    value=float(current_value) if pd.notna(current_value) else float(X_scoring[feature].min()),
                    key=f"input_{feature}"
                )
            else:
                modified_values[feature] = st.text_input(
                    label=feature,
                    value=str(current_value) if pd.notna(current_value) else "",
                    key=f"input_{feature}"
                )

        if st.button("Recalculer le score"):
            all_features = {
                k: (None if pd.isna(v) else v.item() if hasattr(v, 'item') else v)
                for k, v in X_scoring.loc[client_id].items()
            }
            all_features.update(modified_values)
            r = requests.post(f"{BASE_URL}/predict", json={"features": all_features}, timeout=30)
            st.session_state["prediction"] = r.json()
            st.rerun()

        # Infos statiques
        st.header("Informations client")
        st.write(df[df["SK_ID_CURR"] == client_id].T)  # .T pour afficher en colonne

# ── zone principale ───────────────────────────────────────────────────────────
st.title('Tableau de bord - Crédit client')

if "prediction" in st.session_state:
    client_id = st.session_state["selectbox_client_id"]
    prediction = st.session_state["prediction"]
    features_sorted = sorted(
        X_scoring.columns.tolist(),
        key=lambda f: get_global_importance(f, prediction["shap_local"]),
        reverse=True
    )
    st.header("Eligibilité du client")
    if prediction['approved']:
        st.success("✅ Crédit accepté")
        approved = "Accepté"
    else:
        st.error("❌ Crédit refusé")
        approved = "Refusé"
        
    st.write(f"**Client {client_id}** - {approved} — Probabilité de défaut : {prediction['probability']:.2%}")
    with st.expander("ℹ️ Comment lire ce résultat ?"):
        st.write("""
            La jauge indique la **probabilité que le client soit en défaut de paiement**.
            La ligne noire représente le **seuil de décision** : au-delà, le crédit est refusé.
            - 🟢 Zone verte : risque faible
            - 🟠 Zone orange : risque modéré
            - 🔴 Zone rouge : risque élevé
        """)
    display_gauge(prediction["probability"], prediction["threshold"])
    
    st.divider()
    
    st.header("Importance des caractéristiques")
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.write("""
            Ce graphique compare l'importance des principales caractéristiques **pour ce client ** (en orange)
            par rapport à l'importance **moyenne sur tous les clients** (en bleu).
            Plus la barre est longue, plus la caractéristique a influencé la décision.
        """)
    display_shap(prediction["shap_local"], prediction["shap_global"])

    st.divider()

    st.header("Distribution d'une caractéristique")
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.write("""
            L'histogramme montre la **distribution de la caractéristique** sur l'ensemble des clients.
            La **ligne rouge** (ou barre rouge) indique la valeur de ce client,
            permettant de le situer par rapport aux autres.
        """)
    feature = st.selectbox("Choisir une feature :", options=features_sorted, index=0)
    display_distribution(client_id, feature, X_scoring)

    st.divider()

    st.header("Analyse bi-variée (2 caractéristiques)")
    with st.expander("ℹ️ Comment lire ce graphique ?"):
        st.write("""
            Choisir une caractéristique dans la liste.
            Ce graphique croise **deux caractéristiques** pour l'ensemble des clients (points bleus).
            L'**étoile rouge** représente la position de ce client.
            Cela permet de voir si le client est dans une zone typique ou atypique.
        """)
    col1, col2 = st.columns(2)
    with col1:
        feature_x = st.selectbox("Feature X :", options=features_sorted, key="biv_x")
    with col2:
        feature_y = st.selectbox("Feature Y :", options=features_sorted, key="biv_y")
    if st.button("Afficher le graphique"):
        display_bivariate(client_id, feature_x, feature_y, X_scoring)
else:
    st.info("👈 Sélectionnez un client dans la barre latérale pour afficher son tableau de bord.")