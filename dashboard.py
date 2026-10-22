import time
import streamlit as st
from pymongo import MongoClient
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Home Energy - Dashboard Temps Réel",
    page_icon="⚡",
    layout="wide"
)

# ============================================================
# SIDEBAR - CONTRÔLE TEMPS RÉEL
# ============================================================

st.sidebar.title("⚡ Smart Home Live")
st.sidebar.markdown("Architecture Big Data Lambda en temps réel :")
st.sidebar.markdown("- **IoT** : Simulation capteurs maison connectée")
st.sidebar.markdown("- **Kafka** : Ingestion du flux continu")
st.sidebar.markdown("- **Spark MLlib** : Modèle Random Forest (22 features)")
st.sidebar.markdown("- **MongoDB** : Stockage persistant des prédictions")
st.sidebar.divider()

auto_refresh = st.sidebar.checkbox("🔄 Actualisation en direct", value=True)
refresh_interval = st.sidebar.slider("Fréquence d'actualisation (s)", 1, 5, 2)

if st.sidebar.button("Actualiser maintenant"):
    st.rerun()

st.sidebar.divider()
st.sidebar.info("💡 Laissez l'actualisation activée pour voir les données et prédictions défiler en direct.")

# ============================================================
# MONGODB CONNECTION
# ============================================================

@st.cache_resource
def get_mongo_client():
    for uri in ["mongodb://localhost:27017/", "mongodb://mongodb:27017/"]:
        try:
            c = MongoClient(uri, serverSelectionTimeoutMS=2000)
            c.admin.command("ping")
            return c
        except Exception:
            continue
    return MongoClient("mongodb://localhost:27017/")

client = get_mongo_client()
db = client["smart_home_energy"]
collection = db["predictions"]

# ============================================================
# TITLE
# ============================================================

st.title("⚡ Smart Home Energy Dashboard")
st.markdown(
    "Prévision en temps réel de la consommation électrique d'une maison intelligente "
    "avec **Apache Kafka**, **Spark MLlib (Random Forest)** et **MongoDB**."
)

# ============================================================
# LOAD DATA
# ============================================================

documents = list(
    collection.find(
        {},
        {
            "_id": 0,
            "timestamp": 1,
            "current_power": 1,
            "lag_1": 1,
            "lag_24": 1,
            "rolling_mean_3h": 1,
            "rolling_mean_6h": 1,
            "prediction": 1,
            "model": 1
        }
    ).sort("timestamp", 1)
)

# ============================================================
# CHECK DATA
# ============================================================

if not documents:
    st.warning("⏳ Aucune prédiction disponible pour l'instant dans MongoDB. Démarrez le producteur IoT et Spark Streaming.")
else:
    df = pd.DataFrame(documents)

    # Nettoyage affichage timestamp
    df["timestamp_str"] = df["timestamp"].astype(str)

    latest = df.iloc[-1]
    prev_power = df.iloc[-2]["current_power"] if len(df) > 1 else latest["current_power"]
    delta_power = round(latest["current_power"] - prev_power, 3)

    # ========================================================
    # METRICS EN DIRECT
    # ========================================================
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="⚡ Puissance Actuelle (Réelle)",
            value=f"{latest['current_power']:.3f} kW",
            delta=f"{delta_power:+.3f} kW" if len(df) > 1 else None
        )

    with col2:
        st.metric(
            label="🔮 Puissance Prédite (MLlib)",
            value=f"{latest['prediction']:.3f} kW"
        )

    with col3:
        st.metric(
            label="📈 Total Prédictions (MongoDB)",
            value=f"{len(df)} mesures",
            delta=f"+{len(df) - 6}" if len(df) > 6 else None
        )

    with col4:
        st.metric(
            label="🤖 Modèle Actif",
            value="Random Forest",
            delta="40 arbres (prof. 8)"
        )

    # ========================================================
    # CHART ACTUAL VS PREDICTED
    # ========================================================
    st.subheader("📈 Consommation Réelle vs Prédiction en Temps Réel")

    # Affiche les 40 dernières mesures pour un graphique clair et réactif
    chart_data = df.tail(40).copy()
    chart_df = chart_data[["timestamp_str", "current_power", "prediction"]].rename(
        columns={
            "current_power": "Consommation Réelle (kW)",
            "prediction": "Prédiction Random Forest (kW)"
        }
    ).set_index("timestamp_str")

    st.line_chart(chart_df, color=["#1f77b4", "#ff7f0e"])

    # ========================================================
    # TABLE DES DERNIÈRES MESURES
    # ========================================================
    st.subheader("📊 Dernières Prédictions Enregistrées")

    feature_df = df[
        [
            "timestamp_str",
            "current_power",
            "lag_1",
            "lag_24",
            "rolling_mean_3h",
            "rolling_mean_6h",
            "prediction"
        ]
    ].rename(
        columns={
            "timestamp_str": "Horodatage",
            "current_power": "Puissance (kW)",
            "lag_1": "Lag 1 min",
            "lag_24": "Lag 24h",
            "rolling_mean_3h": "Moyenne 3h",
            "rolling_mean_6h": "Moyenne 6h",
            "prediction": "Prédiction (kW)"
        }
    ).tail(15)

    st.dataframe(feature_df, use_container_width=True)

    # ========================================================
    # MODEL INFORMATION
    # ========================================================
    with st.expander("ℹ️ Informations sur le modèle & Performances"):
        st.write(
            "**Modèle** : Random Forest Regression (Spark MLlib) — "
            "40 arbres, profondeur maximale 8, 22 variables explicatives."
        )
        st.write(
            "**Métriques d'évaluation sur jeu de test** : "
            "RMSE : `0.2152` | MAE : `0.0828` | R² : `0.9376` (93.76% de variance expliquée)."
        )

# ============================================================
# AUTO-REFRESH TRIGGER
# ============================================================
if auto_refresh:
    time.sleep(refresh_interval)
    st.rerun()

