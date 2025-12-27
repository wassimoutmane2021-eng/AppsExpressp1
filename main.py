import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# ======================
# 1. CONFIGURATION INITIALE
# ======================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix pour XGBoost sur macOS

# Charger les modèles (avec gestion d'erreur)
try:
    model = joblib.load("models/xgboost_churn_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    label_encoders = joblib.load("models/label_encoders.pkl")
    st.success("✅ Modèles chargés avec succès!")
except Exception as e:
    st.error(f"❌ Erreur de chargement des modèles: {e}")
    st.stop()  # Arrête l'exécution si les modèles ne chargent pas

# ======================
# 2. INTERFACE STREAMLIT
# ======================
st.title("Prédiction de Désabonnement (Churn) - Expresso Telecom")
st.markdown("""
Cette application prédit si un client risque de se désabonner (**Churn = 1**) ou non (**Churn = 0**).
*Remplissez les paramètres du client dans la sidebar.*
""")

# ======================
# 3. FONCTIONS UTILES
# ======================
def decode_label(encoder, value):
    """Décode une valeur encodée en label lisible."""
    try:
        return encoder.inverse_transform([value])[0]
    except:
        return value  # Retourne la valeur brute en cas d'erreur

# ======================
# 4. FORMULAIRE UTILISATEUR (SIDEBAR)
# ======================
st.sidebar.header("Paramètres du Client")

# Variables catégorielles (avec gestion d'erreur)
try:
    region = st.sidebar.selectbox(
        "Région",
        label_encoders['REGION'].classes_,
        index=0
    )
    tenure = st.sidebar.selectbox(
        "Ancienneté (mois)",
        sorted(label_encoders['TENURE'].classes_.astype(int)),
        index=1  # Valeur par défaut: 1 mois
    )
    mrg = st.sidebar.selectbox(
        "MRG (Groupe de Revenue)",
        label_encoders['MRG'].classes_,
        index=0
    )
    top_pack = st.sidebar.selectbox(
        "Forfait Principal",
        label_encoders['TOP_PACK'].classes_,
        index=0
    )
except KeyError as e:
    st.error(f"❌ Clé manquante dans les encodeurs: {e}")
    st.stop()

# Variables numériques
montant = st.sidebar.number_input("Montant dépensé (MONTANT)", min_value=0.0, value=5000.0, step=100.0)
frequence_rech = st.sidebar.number_input("Fréquence de recharge", min_value=0, value=5, step=1)
revenue = st.sidebar.number_input("Revenu mensuel (REVENUE)", min_value=0.0, value=5000.0, step=100.0)
arpu_segment = st.sidebar.number_input("ARPU Segment", min_value=0.0, value=1000.0, step=100.0)
frequence = st.sidebar.number_input("Fréquence d'utilisation (appels/jour)", min_value=0.0, value=10.0, step=1.0)
data_volume = st.sidebar.number_input("Volume de données (Mo)", min_value=0.0, value=1000.0, step=100.0)
on_net = st.sidebar.number_input("Appels On-Net (minutes)", min_value=0.0, value=100.0, step=10.0)
orange = st.sidebar.number_input("Appels vers Orange (minutes)", min_value=0.0, value=50.0, step=10.0)
tigo = st.sidebar.number_input("Appels vers Tigo (minutes)", min_value=0.0, value=10.0, step=1.0)
zone1 = st.sidebar.number_input("Appels Zone 1 (minutes)", min_value=0.0, value=1.0, step=0.1)
zone2 = st.sidebar.number_input("Appels Zone 2 (minutes)", min_value=0.0, value=1.0, step=0.1)
regularity = st.sidebar.number_input("Régularité (jours actifs/mois)", min_value=0.0, value=30.0, step=1.0)
freq_top_pack = st.sidebar.number_input("Fréquence d'utilisation du forfait", min_value=0.0, value=5.0, step=1.0)

# ======================
# 5. PRÉDICTION (BOUTON)
# ======================
if st.sidebar.button("Prédire le Churn"):
    try:
        # Encodage des variables catégorielles
        region_encoded = label_encoders['REGION'].transform([region])[0]
        tenure_encoded = label_encoders['TENURE'].transform([str(tenure)])[0]  # Convertir en str pour l'encoder
        mrg_encoded = label_encoders['MRG'].transform([mrg])[0]
        top_pack_encoded = label_encoders['TOP_PACK'].transform([top_pack])[0]

        # Création du DataFrame (ordre des colonnes = ordre d'entraînement du modèle!)
        input_data = pd.DataFrame({
            'REGION': [region_encoded],
            'TENURE': [tenure_encoded],
            'MONTANT': [montant],
            'FREQUENCE_RECH': [frequence_rech],
            'REVENUE': [revenue],
            'ARPU_SEGMENT': [arpu_segment],
            'FREQUENCE': [frequence],
            'DATA_VOLUME': [data_volume],
            'ON_NET': [on_net],
            'ORANGE': [orange],
            'TIGO': [tigo],
            'ZONE1': [zone1],
            'ZONE2': [zone2],
            'MRG': [mrg_encoded],
            'REGULARITY': [regularity],
            'TOP_PACK': [top_pack_encoded],
            'FREQ_TOP_PACK': [freq_top_pack]
        })

        # Scaling et prédiction
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        # ======================
        # 6. AFFICHAGE DES RÉSULTATS
        # ======================
        st.subheader("Résultat de la Prédiction")
        col1, col2 = st.columns(2)
        with col1:
            if prediction == 1:
                st.error("⚠️ **Risque ÉLEVÉ de désabonnement**")
            else:
                st.success("✅ **Faible risque de désabonnement**")
        with col2:
            st.metric("Probabilité de Churn", f"{probability:.1%}")

        # Détails du client 
        st.subheader("Détails du Client")
        details_df = input_data.copy()
        details_df['REGION'] = decode_label(label_encoders['REGION'], region_encoded)
        details_df['TENURE'] = decode_label(label_encoders['TENURE'], tenure_encoded)
        details_df['MRG'] = decode_label(label_encoders['MRG'], mrg_encoded)
        details_df['TOP_PACK'] = decode_label(label_encoders['TOP_PACK'], top_pack_encoded)
        st.dataframe(details_df.style.highlight_max(axis=0))

    except Exception as e:
        st.error(f"❌ Erreur lors de la prédiction: {e}")
        st.write("Vérifiez que toutes les valeurs sont valides et que les encodeurs correspondent aux données.")
