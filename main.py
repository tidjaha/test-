import streamlit as st
import pandas as pd
import joblib
import os
# Load the trained model
from PIL import Image, UnidentifiedImageError
import gdown
import requests
from io import BytesIO  # Importation nécessaire

# Téléchargez l'image à partir de Google Drive ou OneDrive
url = "https://cloudconvert-files.s3.eu-central-1.amazonaws.com/661669a9-da44-4fd5-ba63-c59c28825284/alitest.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=AKIAI2WCZ54772T33JEQ%2F20250105%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20250105T182150Z&X-Amz-Expires=86400&X-Amz-Signature=a504541d16fd9eac7127f4ae550f135718b505f786a76457936633a06cf1bca7&X-Amz-SignedHeaders=host&response-content-disposition=attachment%3B%20filename%3D%22alitest.jpg%22&response-content-type=image%2Fjpeg&x-id=GetObject"

# Téléchargement de l'image
response = requests.get(url)

if response.status_code == 200:
    try:
        # Convertir le contenu de la réponse en image
        image_data = BytesIO(response.content)
        image = Image.open(image_data)
        image.verify()  # Vérifie si l'image est valide
        st.image(image, caption="Image téléchargée", use_container_width=True)
    except UnidentifiedImageError:
        st.error("Erreur : Le fichier téléchargé n'est pas une image valide.")
    except Exception as e:
        st.error(f"Une erreur est survenue : {e}")
else:
    st.error(f"Erreur : Le téléchargement de l'image a échoué avec le statut {response.status_code}.")

# Chargement des fichiers
model ="model randomforest.pkl"

scaler = joblib.load('scale des notes.pkl')
scaler0= joblib.load("scale du nbslibing.pkl")
encoder0= joblib.load("encoder de la colone ethni.pkl")
encoder1= joblib.load("encoder de la colone parent educ.pkl")
encoder2=joblib.load("encoder de la colone parent marital status.pkl")
encoder3=joblib.load("encoder de la colone practice sport.pkl")
encoder4=joblib.load("encoder de la colone wkly study hours.pkl")



# Define function to make predictions

def predict(input_features):

    # Perform any necessary preprocessing on the input_features

    # Make predictions using the loaded model

    prediction = model.predict(input_features)
    prediction=scaler.inverse_transform(prediction)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.bar(["Math","Lecture","Ecriture"],prediction[0] , color=['red','blue','black'], label=[f"Math = {prediction[0][0]:.2f}",f"Lecture = {prediction[0][1]:.2f}",f"Ecriture = {prediction[0][2]:.2f}"])
    ax.set_ylim(0, 100)
    ax.set_title('Les trois notes')
    ax.legend()
    ax.set_ylabel('Notes')
    plot=st.pyplot(fig)
    return f"votre note de lecture : {prediction[0][1]:.2f}\n\nvotre note d'ecriture :{prediction[0][2]:.2f}\n\nvotre note de math : { prediction[0][0]:.2f}\n\n "

# Create the web interface

def main():

    st.title('modele de prediction des notes des etudiants')

    st.write('remplissez les champs pour avoir la prediction des trois notes')

    # Create input fields for user to enter data

    EthnicGroup = st.selectbox("Ethnie",["group A","group B","group C","group D","group E"])
    encodedEthnicGroup=encoder0.transform([EthnicGroup])[0]

    ParentEduc = st.selectbox("niv educatif des parents",["some high school","high school","some college","associate's degree","bachelor's degree","master's degree"])
    encodedParentEduc=encoder1.transform([ParentEduc])[0]

    ParentMaritalStatus=st.selectbox('situation familliale',["divorced","married","single","widowed"])
    encodedParentMaritalStatus=encoder2.transform([ParentMaritalStatus])[0]

    PracticeSport=st.selectbox('sport',['regularly', 'sometimes' ,'never'])
    encodedPracticeSport=encoder3.transform([PracticeSport])[0]

    NrSiblings=st.number_input("nombre de parent dans l'ecole", min_value=0.0, step=1.0)
    encodedNrSiblings=scaler0.transform([[NrSiblings]])[0][0]

    WklyStudyHours=st.selectbox("nombre d'heure de revision par semaine",['< 5', '5 - 10', '> 10'])
    encodedWklyStudyHours=encoder4.transform([WklyStudyHours])[0]

    Gender=st.selectbox('Genre',['Homme','Femme'])
    if Gender=="Homme":
       encodedGender=True
    else:
       encodedGender=False

    LunchType_standard=st.selectbox('Type de dejeuné',['standard','individuel/maison'])
    if LunchType_standard=='standard':
      encodedLunchType_standard=True
    else:
      encodedLunchType_standard=False

    TestPrep_none=st.selectbox('Preparation test',['Oui','Non'])
    if TestPrep_none=='Oui':
      encodedTestPrep_none=False
    else:
      encodedTestPrep_none=True

    IsFirstChild_yes=st.selectbox('premier enfant',['Oui','Non'])
    if IsFirstChild_yes=='Oui':
      encodedIsFirstChild_yes=True
    else:
      encodedIsFirstChild_yes=False

    TransportMeans_school_bus=st.selectbox("moyen de transport",["Bus de l'ecole","Personnel"])
    if TransportMeans_school_bus=='Bus de l"ecole':
      encodedTransportMeans_school_bus=True
    else:
      encodedTransportMeans_school_bus=False


    # Add more input fields as needed

    # Combine input features into a DataFrame

    input_data = pd.DataFrame({"EthnicGroup":[encodedEthnicGroup],'ParentEduc': [encodedParentEduc],'ParentMaritalStatus': [encodedParentMaritalStatus],
                                "PracticeSport":[encodedPracticeSport],"NrSiblings":[encodedNrSiblings],"WklyStudyHours":[encodedWklyStudyHours],
                               "Gender_male":[encodedGender],"LunchType_standard":[encodedLunchType_standard],
                               "TestPrep_none":[encodedTestPrep_none],"IsFirstChild_yes":[encodedIsFirstChild_yes],
                             "TransportMeans_school_bus":[encodedTransportMeans_school_bus]})

    # Add more features as needed

    if st.button('Predictions'):

        prediction = predict(input_data)

        st.write('Les Predictions sont :\n\n', prediction)

if __name__ == '__main__':

    main()
