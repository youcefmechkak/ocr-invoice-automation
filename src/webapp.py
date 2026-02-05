# Importation des bibliothèques nécessaires
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Évite les conflits de bibliothèques (problèmes liés à Keras/TensorFlow)
import numpy as np  # Pour la manipulation de tableaux et calculs numériques
import streamlit as st  # Framework pour créer une interface web interactive
import re  # Pour utiliser des expressions régulières (recherche de motifs dans le texte)
import tempfile  # Pour gérer des fichiers temporaires
import math  # Pour des calculs mathématiques (ex. comparaison de valeurs)
import json  # Pour sauvegarder/exportation des données au format JSON
import pandas as pd  # Pour manipuler et afficher des données sous forme de tableaux
from pathlib import Path  # Pour manipuler les chemins de fichiers
import cv2  # OpenCV pour le traitement d'images (prétraitement, conversion)
import easyocr  # Bibliothèque OCR pour extraire du texte à partir d'images
import io  # Pour manipuler des flux d'entrée/sortie (ex. sauvegarde d'images)
from pdf2image import convert_from_bytes  # Pour convertir des fichiers PDF en images
import base64  # Pour encoder des fichiers (images, textes) en base64 pour téléchargement
from PIL import Image  # Pour manipuler des images (conversion, sauvegarde)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Invoice Data Extraction",  # Titre de la page web
    page_icon="📊",  # Icône affichée dans l'onglet du navigateur
    layout="wide",  # Mise en page large pour une meilleure utilisation de l'espace
    initial_sidebar_state="collapsed"  # Barre latérale repliée par défaut
)

# CSS personnalisé pour styliser l'interface utilisateur
st.markdown("""
<style>
    .main {
        padding: 2rem 3rem;  /* Ajoute du padding à la zone principale */
    }
    .stButton>button {
        background-color: #4CAF50;  /* Couleur de fond verte pour les boutons */
        color: white;  /* Texte blanc */
        font-weight: bold;  /* Texte en gras */
        padding: 0.5rem 1rem;  /* Padding interne */
        border-radius: 5px;  /* Coins arrondis */
        transition: all 0.3s;  /* Animation fluide lors des interactions */
    }
    .stButton>button:hover {
        background-color: #45a049;  /* Couleur plus foncée au survol */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);  /* Ombre au survol */
    }
    .upload-section {
        border: 2px dashed #ccc;  /* Bordure en pointillés pour la zone de téléchargement */
        border-radius: 10px;  /* Coins arrondis */
        padding: 2rem;  /* Padding interne */
        text-align: center;  /* Texte centré */
        margin-bottom: 2rem;  /* Marge en bas */
    }
    .results-section {
        background-color: #f8f9fa;  /* Fond gris clair pour les résultats */
        border-radius: 10px;  /* Coins arrondis */
        padding: 1.5rem;  /* Padding interne */
        margin-top: 2rem;  /* Marge en haut */
    }
    .download-btn {
        background-color: #007bff !important;  /* Couleur bleue pour les boutons de téléchargement */
    }
    h1 {
        color: #2C3E50;  /* Couleur bleu foncé pour les titres */
    }
    .success-msg {
        color: #4CAF50;  /* Couleur verte pour les messages de succès */
        font-weight: bold;  /* Texte en gras */
    }
    .error-msg {
        color: #f44336;  /* Couleur rouge pour les messages d'erreur */
        font-weight: bold;  /* Texte en gras */
    }
</style>
""", unsafe_allow_html=True)

# Initialisation du lecteur EasyOCR (mis en cache pour éviter de le recharger à chaque exécution)
@st.cache_resource
def load_ocr_reader():
    try:
        import torch
        # Vérification de la disponibilité GPU
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            st.sidebar.success("GPU NVIDIA détecté - Accélération activée")
            try:
                # Essai avec GPU
                return easyocr.Reader(['en', 'fr'], gpu=True)
            except Exception as e:
                st.sidebar.warning(f"Erreur GPU: {str(e)} - Basculé sur CPU")
                return easyocr.Reader(['en', 'fr'], gpu=False)
        else:
            st.sidebar.warning("Aucun GPU détecté - Utilisation du CPU")
            return easyocr.Reader(['en', 'fr'], gpu=False)
            
    except Exception as e:
        st.error(f"Échec d'initialisation OCR: {str(e)}")
        st.stop()
# Variables globales de session pour stocker les résultats et états
if 'extraction_results' not in st.session_state:
    st.session_state.extraction_results = None  # Résultats de l'extraction (données de la facture)
if 'image_preview' not in st.session_state:
    st.session_state.image_preview = None  # Image originale pour prévisualisation
if 'ocr_text' not in st.session_state:
    st.session_state.ocr_text = None  # Texte brut extrait par OCR

# Titre et description de l'application affichés dans l'interface
st.title("📄 Extraction de données de factures")
st.markdown("""
Cette application vous permet d'extraire automatiquement les informations clés de vos factures.
Téléchargez simplement une image (JPG, PNG) ou un PDF de facture et laissez l'application analyser le contenu.
""")

# Section pour téléverser un fichier
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choisissez une image ou un PDF de facture", 
                                type=['jpg', 'jpeg', 'png', 'pdf'],
                                help="Formats acceptés: JPG, PNG, PDF")  # Zone de téléversement pour fichiers image ou PDF
st.markdown("</div>", unsafe_allow_html=True)

### SECTION FONCTIONS DE TRAITEMENT ###

def preprocess_image(image):
    """Prétraite l'image pour améliorer les résultats de l'OCR"""
    # Conversion en niveaux de gris si l'image est en couleur
    if len(image.shape) == 3:  # Vérifie si l'image est en couleur (3 canaux RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convertit en niveaux de gris
    else:
        gray = image  # Si déjà en gris, pas de conversion
    
    # Seuillage adaptatif pour augmenter le contraste
    bin_img = cv2.adaptiveThreshold(
        gray, 255,  # Image source et valeur maximale
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Méthode adaptative basée sur une gaussienne
        cv2.THRESH_BINARY,  # Binarisation (noir ou blanc)
        blockSize=35,  # Taille de la zone pour le calcul du seuil
        C=10  # Constante soustraite au seuil
    )
    
    # Opération morphologique pour réduire le bruit
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # Noyau rectangulaire 2x2
    cleaned = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel, iterations=1)  # Ouvre l'image pour éliminer les petits artefacts
    
    return cleaned  # Retourne l'image prétraitée

def ocr_image(image, reader):
    """
    Exécute l'OCR sur une image prétraitée
    Retourne le texte extrait et l'image prétraitée
    """
    preprocessed = preprocess_image(image)  # Prétraitement de l'image
    
    # Exécute l'OCR avec EasyOCR
    result = reader.readtext(preprocessed, detail=0)  # Extrait le texte sans détails (seulement les chaînes)
    text = "\n".join(result)  # Concatène les résultats avec des sauts de ligne
    
    return text, preprocessed  # Retourne le texte OCR et l'image prétraitée

def extract_invoice_data(text):
    """
    Extrait les données clés de la facture à partir du texte OCR
    Utilise des expressions régulières pour identifier :
    - Numéro de facture
    - Date
    - Client
    - Montants financiers
    """
    # Définir les motifs regex pour chaque champ
    patterns = {
        "invoice_no": r"Invoice no:?\s*(\d+)",  # Numéro de facture (ex. "Invoice no: 123")
        "date": r"Date of issue:?\s*(\d{2}/\d{2}/\d{4})",  # Date (ex. "Date of issue: 12/05/2023")
        "client": r"Client:.*?\n.*?\n(.*?)(?:\n|$)",  # Nom du client (sur une ligne après "Client:")
    }
    
    invoice_data = {}  # Dictionnaire pour stocker les données extraites
    
    # Recherche des motifs dans le texte
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.DOTALL)  # Recherche avec regex, DOTALL pour inclure les sauts de ligne
        if match:
            invoice_data[key] = match.group(1).strip()  # Stocke la valeur trouvée (groupe 1)
        else:
            invoice_data[key] = None  # Si non trouvé, stocke None
    
    # Extraction des montants financiers (net, TVA, total)
    net, vat, gross = extract_financial_values(text)
    invoice_data.update({
        "net_worth": net,  # Montant net
        "vat": vat,  # Montant de la TVA
        "gross_worth": gross  # Montant total
    })
    
    return invoice_data  # Retourne le dictionnaire des données extraites
def extract_financial_values(content):
    """
    Fonction robuste pour extraire les valeurs financières d'une facture.
    Implémente plusieurs stratégies de backup et validation des données.
    """
    # Pattern pour les nombres avec gestion des séparateurs de milliers
    number_pattern = r"[\d\s]+(?:,\d{2})?"
    currency_pattern = r"(?:[\$\€\£]?\s*)?"  # Support des symboles monétaires optionnels
    
    # Tentative avec un pattern strict (3 valeurs sur une ligne)
    strict_pattern = fr"Total.*?{currency_pattern}({number_pattern})\s+{currency_pattern}({number_pattern})\s+{currency_pattern}({number_pattern})"
    match = re.search(strict_pattern, content, re.DOTALL)
    
    if match:
        net, vat, gross = match.groups()
    else:
        # Pattern flexible pour valeurs sur plusieurs lignes
        flexible_pattern = fr"Total\s*[\n\r]+{currency_pattern}({number_pattern})\s*[\n\r]+{currency_pattern}({number_pattern})\s*[\n\r]+{currency_pattern}({number_pattern})"
        match = re.search(flexible_pattern, content, re.DOTALL)
        if match:
            net, vat, gross = match.groups()
        else:
            # Dernier recours: chercher les 3 derniers nombres avant "Top" ou autre marqueur
            last_numbers = re.findall(fr"{currency_pattern}({number_pattern})(?=\s|$)", content)
            if len(last_numbers) >= 3:
                net, vat, gross = last_numbers[-3:]
            else:
                return None, None, None
    
    # Nettoyage et conversion des valeurs
    net = clean_value(net) if 'net' in locals() else None
    vat = clean_value(vat) if 'vat' in locals() else None
    gross = clean_value(gross) if 'gross' in locals() else None
    
    # Calculs de backup si certaines valeurs sont manquantes
    if net is None and vat is not None and gross is not None:
        try:
            net = gross - vat
        except:
            pass
    
    if vat is None and net is not None and gross is not None:
        try:
            vat = gross - net
        except:
            pass
    
    if gross is None and net is not None and vat is not None:
        try:
            gross = net + vat
        except:
            pass
    
    # Validation des résultats
    if None not in [net, vat, gross]:
        if not math.isclose(net + vat, gross, rel_tol=0.01):
            # Correction des inversions potentielles entre VAT et Gross
            if math.isclose(net + gross, vat, rel_tol=0.01):
                vat, gross = gross, vat
    
    return net, vat, gross
def clean_value(val):
    """
    Fonction utilitaire pour nettoyer et convertir une valeur numérique en float.
    Corrige les erreurs courantes d'OCR et normalise le format des nombres.
    """
    if val is None:
        return None

    # Dictionnaire des corrections pour les erreurs d'OCR typiques
    corrections = {
        'B': '8',  # Remplace 'B' par '8'
        'l': '1',  # Remplace 'l' par '1'
        'I': '1',  # Remplace 'I' par '1'
        'O': '0',  # Remplace 'O' par '0'
        ' ': ''   # Supprime les espaces
    }

    try:
        # Applique les corrections caractère par caractère
        cleaned = ''.join([corrections.get(c, c) for c in val])
        # Normalise les séparateurs : remplace ',' par '.' et supprime les espaces
        cleaned = cleaned.replace(' ', '').replace(',', '.')
        return float(cleaned)  # Convertit en float
    except (ValueError, AttributeError):
        return None  # Retourne None si la conversion échoue

def validate_results(invoice_data):
    """Valide les résultats extraits et calcule les valeurs manquantes si possible"""
    net = invoice_data.get("net_worth")  # Montant net
    vat = invoice_data.get("vat")  # TVA
    gross = invoice_data.get("gross_worth")  # Total
    
    # Calcule les valeurs manquantes si possible
    if net is None and vat is not None and gross is not None:
        try:
            net = gross - vat  # Calcule le net
            invoice_data["net_worth"] = net
        except:
            pass
    
    if vat is None and net is not None and gross is not None:
        try:
            vat = gross - net  # Calcule la TVA
            invoice_data["vat"] = vat
        except:
            pass
    
    if gross is None and net is not None and vat is not None:
        try:
            gross = net + vat  # Calcule le total
            invoice_data["gross_worth"] = gross
        except:
            pass
    
    # Vérifie la cohérence des valeurs
    if None not in [net, vat, gross]:
        if not math.isclose(net + vat, gross, rel_tol=0.01):  # Vérifie si net + TVA ≈ total
            # Corrige les erreurs évidentes (ex. inversion TVA/total)
            if math.isclose(net + gross, vat, rel_tol=0.01):  # Si net + total ≈ TVA
                invoice_data["vat"] = gross  # Inverse les valeurs
                invoice_data["gross_worth"] = vat
    
    return invoice_data  # Retourne les données validées

def convert_pdf_to_image(pdf_file):
    """Convertit la première page d'un PDF en image"""
    pdf_bytes = pdf_file.read()  # Lit le fichier PDF en bytes
    images = convert_from_bytes(pdf_bytes, dpi=200)  # Convertit en images avec une résolution de 200 DPI
    if images:
        # Convertit l'image PIL en format OpenCV (BGR)
        open_cv_image = cv2.cvtColor(np.array(images[0]), cv2.COLOR_RGB2BGR)
        return open_cv_image
    return None  # Retourne None si la conversion échoue

def get_image_download_link(img, filename, text):
    """Génère un lien de téléchargement pour une image"""
    buffered = io.BytesIO()  # Crée un buffer pour stocker l'image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convertit l'image en RGB pour PIL
    Image.fromarray(img).save(buffered, format="PNG")  # Sauvegarde l'image dans le buffer
    img_str = base64.b64encode(buffered.getvalue()).decode()  # Encode en base64
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{text}</a>'  # Crée le lien HTML
    return href

def get_file_download_link(data, filename, text):
    """Génère un lien de téléchargement pour un fichier texte"""
    b64 = base64.b64encode(data.encode()).decode()  # Encode les données en base64
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'  # Crée le lien HTML
    return href

# Bouton pour lancer l'analyse
if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])  # Crée deux colonnes pour organiser l'interface
    
    with col1:
        if st.button("🔍 Analyser le document", key="analyze_btn"):  # Bouton pour lancer l'analyse
            with st.spinner('Traitement en cours...'):  # Affiche un indicateur de chargement
                try:
                    # Charge le lecteur OCR
                    reader = load_ocr_reader()
                    
                    # Traite selon le type de fichier
                    file_extension = Path(uploaded_file.name).suffix.lower()  # Récupère l'extension du fichier
                    
                    if file_extension == '.pdf':
                        # Convertit le PDF en image
                        image = convert_pdf_to_image(uploaded_file)
                        if image is None:
                            st.error("Impossible de convertir le PDF en image.")  # Affiche une erreur si échec
                            st.stop()  # Arrête l'exécution
                    else:
                        # Lit l'image directement
                        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)  # Décode l'image en format OpenCV
                    
                    # Sauvegarde l'image pour prévisualisation
                    preview_img = image.copy()
                    st.session_state.image_preview = preview_img
                    
                    # Exécute l'OCR
                    ocr_text, preprocessed_img = ocr_image(image, reader)
                    st.session_state.ocr_text = ocr_text  # Stocke le texte OCR
                    
                    # Extrait les données de la facture
                    invoice_data = extract_invoice_data(ocr_text)
                    
                    # Valide et corrige les données
                    invoice_data = validate_results(invoice_data)
                    
                    # Stocke les résultats
                    st.session_state.extraction_results = invoice_data
                    
                    # Affiche un message de succès
                    st.success("Extraction terminée avec succès!")
                
                except Exception as e:
                    st.error(f"Une erreur s'est produite lors de l'analyse: {str(e)}")  # Affiche l'erreur si échec

    # Affiche les options supplémentaires
    if uploaded_file is not None:
        with col2:
            st.markdown("### Options supplémentaires")
            show_ocr = st.checkbox("Afficher le texte OCR brut", value=False)  # Case pour afficher le texte OCR
            show_debug = st.checkbox("Mode debug", value=False)  # Case pour activer le mode debug

# Affiche les résultats si disponibles
if st.session_state.extraction_results:
    st.markdown("<div class='results-section'>", unsafe_allow_html=True)
    st.subheader("📋 Résultats de l'extraction")
    
    # Formatte les résultats pour l'affichage
    results = st.session_state.extraction_results
    formatted_results = {
        "Numéro de Facture": results.get("invoice_no", "Non détecté"),  # Numéro de facture ou message par défaut
        "Date": results.get("date", "Non détectée"),  # Date ou message par défaut
        "Client": results.get("client", "Non détecté"),  # Client ou message par défaut
        "Montant Net": f"{results.get('net_worth', 'Non détecté')} $" if results.get('net_worth') is not None else "Non détecté",
        "TVA": f"{results.get('vat', 'Non détectée')} $" if results.get('vat') is not None else "Non détectée",
        "Total": f"{results.get('gross_worth', 'Non détecté')} $" if results.get('gross_worth') is not None else "Non détecté"
    }
    
    # Affiche les résultats sous forme de tableau
    df = pd.DataFrame([formatted_results])
    st.dataframe(df.T.rename(columns={0: "Valeur"}), use_container_width=True)  # Affiche le tableau transposé
    
    # Vérifie la cohérence des valeurs financières
    net = results.get("net_worth")
    vat = results.get("vat")
    gross = results.get("gross_worth")
    
    if None not in [net, vat, gross]:
        if math.isclose(net + vat, gross, rel_tol=0.01):  # Vérifie si net + TVA ≈ total
            st.markdown("<p class='success-msg'>✅ Les valeurs financières sont cohérentes</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='error-msg'>⚠️ Les valeurs financières ne sont pas cohérentes</p>", unsafe_allow_html=True)
    else:
        missing = []  # Liste des valeurs manquantes
        if net is None:
            missing.append("Montant Net")
        if vat is None:
            missing.append("TVA")
        if gross is None:
            missing.append("Total")
        
        st.markdown(f"<p class='error-msg'>⚠️ Valeurs manquantes: {', '.join(missing)}</p>", unsafe_allow_html=True)
    
    # Options de téléchargement
    st.subheader("📥 Télécharger les résultats")
    col1, col2 = st.columns(2)
    
    with col1:
        # Téléchargement au format JSON
        json_data = json.dumps(results, indent=4, ensure_ascii=False)  # Convertit les résultats en JSON
        st.download_button(
            label="Télécharger en JSON",
            data=json_data,
            file_name="invoice_data.json",
            mime="application/json",
            key="json_download",
            help="Télécharger les résultats au format JSON"
        )
    
    with col2:
        # Téléchargement au format CSV
        csv_df = pd.DataFrame([results])
        csv = csv_df.to_csv(index=False)  # Convertit les résultats en CSV
        st.download_button(
            label="Télécharger en CSV",
            data=csv,
            file_name="invoice_data.csv",
            mime="text/csv",
            key="csv_download",
            help="Télécharger les résultats au format CSV"
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Affiche les informations de débogage si activé
    if show_debug and 'show_debug' in locals():
        st.subheader("🔧 Informations de débogage")
        
        # Affiche l'image originale
        if st.session_state.image_preview is not None:
            st.image(cv2.cvtColor(st.session_state.image_preview, cv2.COLOR_BGR2RGB), 
                     caption="Image originale", use_column_width=True)
        
        # Affiche le texte OCR brut si demandé
        if st.session_state.ocr_text is not None and show_ocr:
            st.subheader("Texte OCR brut")
            st.text_area("Texte OCR brut", st.session_state.ocr_text, height=300, key="ocr_text_area")

# Affiche le texte OCR brut si la case est cochée
if 'show_ocr' in locals() and show_ocr and st.session_state.ocr_text:
    st.subheader("📝 Texte OCR extrait")
    st.text_area("Texte OCR brut", st.session_state.ocr_text, height=300, key="ocr_text_area-1")

# Pied de page
st.markdown("---")
st.markdown("""
<div style="text-align: center">
    <p>© 2025 - Extraaction M1-ASD 24/25 </p>
</div>
""", unsafe_allow_html=True)

# Importations supplémentaires (redondantes, déjà incluses plus haut)
import numpy as np
from PIL import Image