import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Charger les modèles et les encodeurs
model = joblib.load('model randomforest.pkl')
scaler = joblib.load('scale des notes.pkl')
scaler0 = joblib.load("scale du nbslibing.pkl")
encoder0 = joblib.load("encoder de la colone ethni.pkl")
encoder1 = joblib.load("encoder de la colone parent educ.pkl")
encoder2 = joblib.load("encoder de la colone parent marital status.pkl")
encoder3 = joblib.load("encoder de la colone practice sport.pkl")
encoder4 = joblib.load("encoder de la colone wkly study hours.pkl")

# Fonction de prédiction
def predict(input_features):
    prediction = model.predict(input_features)
    prediction = scaler.inverse_transform(prediction)
    
    # Créer un graphique pour les résultats
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Math", "Lecture", "Écriture"], prediction[0], 
           color=['red', 'blue', 'black'], 
           label=[f"Math = {prediction[0][0]:.2f}", 
                  f"Lecture = {prediction[0][1]:.2f}", 
                  f"Écriture = {prediction[0][2]:.2f}"])
    ax.set_ylim(0, 100)
    ax.set_title('Les trois notes')
    ax.legend()
    ax.set_ylabel('Notes')
    st.pyplot(fig)
    
    return (f"Votre note de lecture : {prediction[0][1]:.2f}\n"
            f"Votre note d'écriture : {prediction[0][2]:.2f}\n"
            f"Votre note de math : {prediction[0][0]:.2f}")

# Interface utilisateur avec Streamlit
def main():
    st.title('Modèle de prédiction des notes des étudiants')
    st.write('Remplissez les champs pour obtenir les prédictions des trois notes')

    # Champs de saisie utilisateur
    EthnicGroup = st.selectbox("Ethnie", ["group A", "group B", "group C", "group D", "group E"])
    encodedEthnicGroup = encoder0.transform([EthnicGroup])[0]

    ParentEduc = st.selectbox("Niveau éducatif des parents", 
                              ["some high school", "high school", "some college", 
                               "associate's degree", "bachelor's degree", "master's degree"])
    encodedParentEduc = encoder1.transform([ParentEduc])[0]

    ParentMaritalStatus = st.selectbox('Situation familiale', 
                                       ["divorced", "married", "single", "widowed"])
    encodedParentMaritalStatus = encoder2.transform([ParentMaritalStatus])[0]

    PracticeSport = st.selectbox('Pratique du sport', ['regularly', 'sometimes', 'never'])
    encodedPracticeSport = encoder3.transform([PracticeSport])[0]

    NrSiblings = st.number_input("Nombre de frères et sœurs", min_value=0.0, step=1.0)
    encodedNrSiblings = scaler0.transform([[NrSiblings]])[0][0]

    WklyStudyHours = st.selectbox("Nombre d'heures de révision par semaine", ['< 5', '5 - 10', '> 10'])
    encodedWklyStudyHours = encoder4.transform([WklyStudyHours])[0]

    Gender = st.selectbox('Genre', ['Homme', 'Femme'])
    encodedGender = True if Gender == "Homme" else False

    LunchType_standard = st.selectbox('Type de déjeuner', ['standard', 'individuel/maison'])
    encodedLunchType_standard = True if LunchType_standard == 'standard' else False

    TestPrep_none = st.selectbox('Préparation au test', ['Oui', 'Non'])
    encodedTestPrep_none = False if TestPrep_none == 'Oui' else True

    IsFirstChild_yes = st.selectbox('Premier enfant', ['Oui', 'Non'])
    encodedIsFirstChild_yes = True if IsFirstChild_yes == 'Oui' else False

    TransportMeans_school_bus = st.selectbox("Moyen de transport", ["Bus de l'école", "Personnel"])
    encodedTransportMeans_school_bus = True if TransportMeans_school_bus == "Bus de l'école" else False

    # Combiner les données saisies
    input_data = pd.DataFrame({
        "EthnicGroup": [encodedEthnicGroup],
        "ParentEduc": [encodedParentEduc],
        "ParentMaritalStatus": [encodedParentMaritalStatus],
        "PracticeSport": [encodedPracticeSport],
        "NrSiblings": [encodedNrSiblings],
        "WklyStudyHours": [encodedWklyStudyHours],
        "Gender_male": [encodedGender],
        "LunchType_standard": [encodedLunchType_standard],
        "TestPrep_none": [encodedTestPrep_none],
        "IsFirstChild_yes": [encodedIsFirstChild_yes],
        "TransportMeans_school_bus": [encodedTransportMeans_school_bus]
    })

    if st.button('Prédictions'):
        prediction = predict(input_data)
        st.write('Les prédictions sont :', prediction)

if __name__ == '__main__':
    main()
