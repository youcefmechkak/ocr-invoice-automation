# Invoice Extraction Pipeline

An automated system to extract structured data from scanned invoices and images using Optical Character Recognition (OCR) and Regular Expressions (Regex). This project converts unstructured visual data into a queryable SQLite database and exportable JSON/CSV formats.

## Project Overview
Manual data entry from invoices is slow and error-prone. This project provides an end-to-end Python pipeline to:
1.  **Process** images and PDF documents.
2.  **Clean** documents using computer vision to improve text readability.
3.  **Extract** raw text via OCR.
4.  **Parse** specific fields (Invoice #, Date, Client, Totals) using Regex.
5.  **Store & Visualize** data through a database and a web-based dashboard.



## Key Features
* **Automatic Preprocessing:** Applies binarization and noise removal via OpenCV.
* **Intelligent Parsing:** Uses Regular Expressions to identify financial patterns and dates.
* **Data Validation:** Logic to verify totals (Net + VAT = Gross).
* **Interactive UI:** Built with Streamlit to allow users to upload invoices and see results instantly.
* **Structured Storage:** Saves all historical extractions into a SQLite database.

## Tech Stack
* **OCR Engine:** EasyOCR
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV (cv2), PIL, pdf2image
* **Data Management:** Pandas, SQLite3, JSON, CSV
* **Logic:** Python Regular Expressions (re)

## Repository Structure
```text
/ocr-invoice-automation
│
├── data/
│   ├── raw/                
│   └── processed/          
│
├── src/                    
│   ├── ocr_pipleine_flow.ipynb   
│   ├── processor.py           
│   └── webapp.py                      
│
├── database/
│   └── Extraction.sqlite3        
│
├── report/
|   └── report.pdf
|     
└── README.md
```

## Installation
Clone the repository:

```bash
git clone https://github.com/your-username/invoice-extraction-pipeline.git
cd invoice-extraction-pipeline
```
Install dependencies:

```bash
pip install streamlit easyocr pandas opencv-python pdf2image
```

Note: If processing PDFs, ensure Poppler is installed on your system.

## Usage
**Running the Web Interface**    
To launch the interactive dashboard where you can upload and test individual invoices:

```bash
streamlit run src/webapp.py
```
**Batch Processing**    
To process all OCR text files in a directory and save them to the database:
1. Ensure your .txt files from the OCR step are in the designated input folder.
2. Run the processor script:

```bash
python src/processor.py
```

## Data Flow
1. **Input**: Scanned image or PDF.
2. **Vision**: OpenCV cleans the image (thresholding/denoising).
3. **OCR**: EasyOCR converts pixels to raw text strings.
4. **Regex**: The system identifies: Invoice ID, Date, Client Name, Net Worth, VAT, and Total.
5. **Output**: Data is committed to invoice_records.db and exported to invoices.json.  
