import streamlit as st
import requests

# Titre de l'application
st.title("Prédiction du diabète avec un modèle ML")

# Description
st.write("""
    Cette application prédit si un patient a un diabète ou non basé sur les caractéristiques suivantes :
    - Nombre de grossesses
    - Concentration en glucose
    - Pression artérielle
    - Épaisseur de la peau
    - Résultats des tests de grossesse
    - Indice de masse corporelle (IMC)
    - Antécédents familiaux de diabète
    - Âge
""")

# Saisie des données par l'utilisateur
preg = st.number_input('Nombre de grossesses', min_value=0.0)
plas = st.number_input('Concentration en glucose', min_value=0.0)
pres = st.number_input('Pression artérielle', min_value=0.0)
skin = st.number_input('Épaisseur de la peau', min_value=0.0)
test = st.number_input('Résultats des tests de grossesse', min_value=0.0)
mass = st.number_input('Indice de masse corporelle', min_value=0.0)
pedi = st.number_input('Antécédents familiaux de diabète', min_value=0.0)
age = st.number_input('Âge', min_value=0)

# Préparer les données d'entrée
data = {
    "preg": preg,
    "plas": plas,
    "pres": pres,
    "skin": skin,
    "test": test,
    "mass": mass,
    "pedi": pedi,
    "age": age
}

# Appel à l'API de prédiction (via POST)
if st.button('Faire la prédiction'):
    # Faire la requête à ton API FastAPI
    url = "http://127.0.0.1:8000/predict/"
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        prediction = response.json()
        st.write(f"Prédiction : {prediction['prediction']}")
    else:
        st.error("Erreur lors de la prédiction.")




