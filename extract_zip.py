import zipfile
import os

def extract_zip(file_path, extract_to):
    """
    Décompresse un fichier ZIP dans un répertoire spécifié.
    """
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Fichier décompressé dans : {extract_to}")

# Exemple d'utilisation
extract_zip('resources/data.zip', 'extracted_data/')
