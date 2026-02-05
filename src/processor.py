import re
import os
import sqlite3
import json
import csv
import pandas as pd
import math

# Configuration des chemins d'accès aux différents dossiers
directory = r"C:\Users\M.Ben\Desktop\EI\output3_easyocr"      # Dossier source des factures
output_dir = r"C:\Users\M.Ben\Desktop\EI\Extraction"          # Dossier pour les exports
db_path = r"C:\Users\M.Ben\Desktop\EI\Extraction.sqlite3"     # Chemin de la base de données SQLite
debug_dir = r"C:\Users\M.Ben\Desktop\EI\Debug"               # Dossier pour les fichiers de débogage

# Création des dossiers nécessaires si ils n'existent pas
for dir_path in [output_dir, debug_dir]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# Établissement de la connexion à la base de données SQLite
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Création de la table des factures avec ses champs spécifiés

cursor.execute("""
CREATE TABLE IF NOT EXISTS invoices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,    
    filename TEXT,                           
    invoice_no TEXT,                         
    date TEXT,                             
    client TEXT,                             
    net_worth REAL,                          
    vat REAL,                                
    gross_worth REAL                        ´´
)
""")
conn.commit()

# Dictionnaire des motifs de reconnaissance pour l'extraction des données
patterns = {
    "invoice_no": r"Invoice no:?\s*(\d+)",           # Capture du numéro de facture
    "date": r"Date of issue:?\s*(\d{2}/\d{2}/\d{4})", # Capture de la date au format JJ/MM/AAAA
    "client": r"Client:.*?\n.*?\n(.*?)(?:\n|$)",       # Capture du nom du client
    "financial_section": r"SUMMARY.*?Total\s+([\d\s,.]+)\s+([\d\s,.]+)\s+([\d\s,.]+)"  # Section financière
}

# Liste de motifs alternatifs pour extraire les valeurs financières
financial_patterns = [
    # Pattern principal: Total dans la section SUMMARY
    r"SUMMARY.*?Total\s+([\d\s,.]+)\s+([\d\s,.]+)\s+([\d\s,.]+)",
    # Patterns alternatifs
    r"Net worth\s*\n\s*([\d\s,.]+).*?VAT\s*\n\s*([\d\s,.]+).*?Gross worth\s*\n\s*([\d\s,.]+)",
    r"Net worth:?\s*([\d\s,.]+).*?VAT:?\s*([\d\s,.]+).*?Gross worth:?\s*([\d\s,.]+)",
    r"Total\s+([\d\s,.]+)\s+([\d\s,.]+)\s+([\d\s,.]+)"
]

# Variables globales pour stocker les résultats
all_invoices = []              # Liste de toutes les factures traitées
problematic_files = []         # Liste des fichiers présentant des problèmes

def extract_financial_values(content, filename):
    """
    Fonction robuste pour extraire les valeurs financières d'une facture.
    Implémente plusieurs stratégies de backup et validation des données.
    """
    debug_info = f"=== Débogage pour {filename} ===\n"
    debug_info += f"Contenu complet:\n{content}\n\n"
    debug_info += "Tentatives d'extraction des valeurs financières:\n"

    # Pattern pour les nombres avec gestion des séparateurs de milliers
    number_pattern = r"[\d\s]+(?:,\d{2})?"
    currency_pattern = r"(?:[\$\€\£]?\s*)?"  # Support des symboles monétaires optionnels
    
    # Tentative avec un pattern strict (3 valeurs sur une ligne)
    strict_pattern = fr"Total.*?{currency_pattern}({number_pattern})\s+{currency_pattern}({number_pattern})\s+{currency_pattern}({number_pattern})"
    match = re.search(strict_pattern, content, re.DOTALL)
    
    if match:
        net, vat, gross = match.groups()
        debug_info += f"Match strict: net_raw='{net}', vat_raw='{vat}', gross_raw='{gross}'\n"
    else:
        # Pattern flexible pour valeurs sur plusieurs lignes
        flexible_pattern = fr"Total\s*[\n\r]+{currency_pattern}({number_pattern})\s*[\n\r]+{currency_pattern}({number_pattern})\s*[\n\r]+{currency_pattern}({number_pattern})"
        match = re.search(flexible_pattern, content, re.DOTALL)
        if match:
            net, vat, gross = match.groups()
            debug_info += f"Match flexible: net_raw='{net}', vat_raw='{vat}', gross_raw='{gross}'\n"
        else:
            # Dernier recours: chercher les 3 derniers nombres avant "Top" ou autre marqueur
            last_numbers = re.findall(fr"{currency_pattern}({number_pattern})(?=\s|$)", content)
            if len(last_numbers) >= 3:
                net, vat, gross = last_numbers[-3:]
                debug_info += f"Match derniers nombres: net_raw='{net}', vat_raw='{vat}', gross_raw='{gross}'\n"
    
    # Nettoyage et conversion des valeurs
    net = clean_value(net) if 'net' in locals() else None
    vat = clean_value(vat) if 'vat' in locals() else None
    gross = clean_value(gross) if 'gross' in locals() else None
    
    # Calculs de backup si certaines valeurs sont manquantes
    if net is None and vat is not None and gross is not None:
        try:
            net = gross - vat
            debug_info += f"Net calculé: {gross} - {vat} = {net}\n"
        except:
            pass
    
    if vat is None and net is not None and gross is not None:
        try:
            vat = gross - net
            debug_info += f"VAT calculé: {gross} - {net} = {vat}\n"
        except:
            pass
    
    if gross is None and net is not None and vat is not None:
        try:
            gross = net + vat
            debug_info += f"Gross calculé: {net} + {vat} = {gross}\n"
        except:
            pass
    
    # Validation des résultats
    if None not in [net, vat, gross]:
        if not math.isclose(net + vat, gross, rel_tol=0.01):
            debug_info += f"⚠️ Validation échouée: {net} + {vat} = {net+vat} ≠ {gross}\n"
            # Correction des inversions potentielles entre VAT et Gross
            if math.isclose(net + gross, vat, rel_tol=0.01):
                vat, gross = gross, vat
                debug_info += f"Correction inversion VAT/Gross: {vat}, {gross}\n"
    
    debug_info += f"Valeurs finales: net={net}, vat={vat}, gross={gross}\n"
    return net, vat, gross, debug_info

def clean_value(val):
    """
    Fonction utilitaire pour nettoyer et convertir une valeur numérique.
    Corrige les erreurs courantes d'OCR et normalise le format.
    """
    if val is None:
        return None
    
    # Dictionnaire des corrections pour les erreurs d'OCR typiques
    corrections = {
        'B': '8',
        'l': '1',
        'I': '1',
        'O': '0',
        ' ': ''
    }
    
    try:
        # Application des corrections et nettoyage
        cleaned = ''.join([corrections.get(c, c) for c in val])
        cleaned = cleaned.replace(' ', '').replace(',', '.')
        return float(cleaned)
    except (ValueError, AttributeError):
        return None

# Traitement des fichiers de factures
for filename in os.listdir(directory):
    if filename.endswith(".txt"):
        file_path = os.path.join(directory, filename)
        
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            invoice_data = {"filename": filename}
            
            # Extraction des informations basiques de la facture
            for key, pattern in patterns.items():
                if key == "financial_section":
                    continue
                    
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    invoice_data[key] = match.group(1).strip()
                else:
                    invoice_data[key] = None
            
            # Extraction et validation des valeurs financières
            net, vat, gross, debug_info = extract_financial_values(content, filename)
            invoice_data.update({
                "net_worth": net,
                "vat": vat,
                "gross_worth": gross
            })

            # Gestion des fichiers problématiques
            if None in [net, vat, gross]:
                problematic_files.append(filename)
                debug_file_path = os.path.join(debug_dir, f"debug_{filename}")
                with open(debug_file_path, "w", encoding="utf-8") as debug_file:
                    debug_file.write(debug_info)
                print(f"⚠️ Valeurs financières manquantes dans {filename}, débogage enregistré.")
            else:
                # Validation supplémentaire
                expected_gross = net + vat
                if not math.isclose(expected_gross, gross, rel_tol=0.01):
                    print(f"⚠️ Alerte validation pour {filename}: {net} + {vat} = {expected_gross} ≠ {gross}")

            all_invoices.append(invoice_data)
            
            # Sauvegarde en base de données
            try:
                cursor.execute("""
                INSERT INTO invoices (
                    filename, invoice_no, date, client, 
                    net_worth, vat, gross_worth
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    invoice_data["filename"],
                    invoice_data["invoice_no"],
                    invoice_data["date"],
                    invoice_data["client"],
                    invoice_data["net_worth"],
                    invoice_data["vat"],
                    invoice_data["gross_worth"]
                ))
                conn.commit()
                status = "✅" if None not in [net, vat, gross] else "❌"
                print(f"{status} Traité: {filename} => Net: {net}, VAT: {vat}, Gross: {gross}")
            except sqlite3.Error as sql_err:
                print(f"❌ Erreur de base de données avec {filename}: {sql_err}")
                conn.rollback()
        
        except Exception as e:
            print(f"❌ Échec du traitement de {filename}: {e}")
            continue

# Génération des exports et rapports
if problematic_files:
    problem_summary_path = os.path.join(debug_dir, "problematic_files_summary.txt")
    with open(problem_summary_path, "w", encoding="utf-8") as summary_file:
        summary_file.write(f"Nombre de fichiers problématiques: {len(problematic_files)}\n\n")
        for i, filename in enumerate(problematic_files, 1):
            summary_file.write(f"{i}. {filename}\n")
    print(f"\n⚠️ {len(problematic_files)} fichiers ont des valeurs financières manquantes.")
    print(f"Résumé enregistré dans: {problem_summary_path}")

# Export vers JSON
json_path = os.path.join(output_dir, "invoices.json")
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(all_invoices, json_file, indent=4, ensure_ascii=False)

# Export vers CSV
csv_path = os.path.join(output_dir, "invoices.csv")
with open(csv_path, 'w', encoding='utf-8', newline='') as csv_file:
    fieldnames = ["filename", "invoice_no", "date", "client", 
                 "net_worth", "vat", "gross_worth"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    writer.writeheader()
    for invoice in all_invoices:
        csv_row = {
            "filename": invoice["filename"],
            "invoice_no": invoice["invoice_no"],
            "date": invoice["date"],
            "client": invoice["client"],
            "net_worth": invoice["net_worth"],
            "vat": invoice["vat"],
            "gross_worth": invoice["gross_worth"]
        }
        writer.writerow(csv_row)

# Génération du tableau HTML
df = pd.DataFrame(all_invoices)
html_path = os.path.join(output_dir, "invoices_table.html")
try:
    html_content = df.to_html(index=False, na_rep="N/A")
    with open(html_path, 'w', encoding='utf-8') as html_file:
        html_file.write("""
        <html>
        <head>
            <title>Invoice Data</title>
            <style>
                table { border-collapse: collapse; width: 100%; }
                th, td { text-align: left; padding: 8px; border: 1px solid #ddd; }
                tr:nth-child(even) { background-color: #f2f2f2; }
                th { background-color: #4CAF50; color: white; }
                .problem { background-color: #ffdddd; }
            </style>
        </head>
        <body>
            <h2>Extracted Invoice Data</h2>
        """ + html_content + """
        </body>
        </html>
        """)
    print(f"✅ Tableau HTML généré: {html_path}")
except Exception as e:
    print(f"❌ Erreur lors de la génération du tableau HTML: {e}")

# Génération du rapport de débogage global
debug_report_path = os.path.join(debug_dir, "debug_report.txt")
with open(debug_report_path, 'w', encoding='utf-8') as report_file:
    report_file.write("RAPPORT DE DÉBOGAGE DE L'EXTRACTION DE FACTURES\n")
    report_file.write("=" * 50 + "\n\n")
    
    report_file.write(f"Nombre total de fichiers traités: {len(all_invoices)}\n")
    report_file.write(f"Nombre de fichiers problématiques: {len(problematic_files)}\n")
    report_file.write(f"Taux de réussite: {100 - (len(problematic_files)/len(all_invoices)*100):.2f}%\n\n")
    
    report_file.write("DISTRIBUTION DES ERREURS:\n")
    report_file.write("-" * 30 + "\n")
    
    error_types = {
        "net_only": 0,
        "vat_only": 0,
        "gross_only": 0,
        "net_vat": 0,
        "net_gross": 0,
        "vat_gross": 0,
        "all_values": 0
    }
    
    for invoice in all_invoices:
        net = invoice["net_worth"]
        vat = invoice["vat"]
        gross = invoice["gross_worth"]
        
        if net is None and vat is None and gross is None:
            error_types["all_values"] += 1
        elif net is None and vat is None:
            error_types["net_vat"] += 1
        elif net is None and gross is None:
            error_types["net_gross"] += 1
        elif vat is None and gross is None:
            error_types["vat_gross"] += 1
        elif net is None:
            error_types["net_only"] += 1
        elif vat is None:
            error_types["vat_only"] += 1
        elif gross is None:
            error_types["gross_only"] += 1
    
    for error_type, count in error_types.items():
        if count > 0:
            report_file.write(f"{error_type}: {count} fichiers\n")

# Fermeture de la connexion SQLite
conn.close()

print(f"\n✅ Extraction et stockage des données terminés avec succès!")
print(f"Résultats exportés vers:")
print(f"- Base de données: {db_path}")
print(f"- JSON: {json_path}")
print(f"- CSV: {csv_path}")
print(f"- HTML: {html_path}")
print(f"- Rapport de débogage: {debug_report_path}")
print(f"- Fichiers de débogage individuels: {debug_dir}")