import os
import json
from preprocess import ResumeProcessor

'''
Input: 
Path of parent folder contains subfolders including PDFs
Path of output folder

Output:
Folder(contains subfolders) of text files converted from PDF files
'''
with open("config.json", "r") as config_file:
    config = json.load(config_file)
tesseract_path = config.get("tesseract_path", "tesseract")
processor = ResumeProcessor(tesseract_path=tesseract_path)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PDF_FOLDER = os.path.join(BASE_DIR,"cvs_data")
OUTPUT_FOLDER = os.path.join(BASE_DIR,"cvs_data_text")

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

def generate_text_folder(pdf_folder, output_folder):
    for root, _, files in os.walk(pdf_folder):
        for filename in files:
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(root, filename)  # Full PDF path
                relative_path = os.path.relpath(root, pdf_folder)  # Get relative path
                output_subfolder = os.path.join(output_folder, relative_path)  # Create matching output subfolder

                os.makedirs(output_subfolder, exist_ok=True)

                txt_filename = os.path.splitext(filename)[0] + ".txt"  # Change extension to .txt
                txt_path = os.path.join(output_subfolder, txt_filename)

                print(f"Processing: {pdf_path}...")

                try:
                    extracted_text = processor.process_resume(pdf_path)  # Extract text
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)  # Save text to file
                    
                    print(f"Saved: {txt_path}")
                except Exception as e:
                    print(f"Error processing {pdf_path}: {e}")

#generate_text_folder(MAIN_PDF_FOLDER, OUTPUT_FOLDER)

print("All PDFs processed successfully!")