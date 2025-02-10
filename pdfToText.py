import os
from preprocess import ResumeProcessor

MAIN_PDF_FOLDER = "D:\datasets\cvs_data"
OUTPUT_FOLDER = "D:\datasets\cvs_data_text"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

processor = ResumeProcessor(tesseract_path=r"C:\Program Files\Tesseract-OCR\tesseract.exe")  # Update path if needed

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

                print(f"📄 Processing: {pdf_path}...")

                try:
                    extracted_text = processor.process_resume(pdf_path)  # Extract text
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(extracted_text)  # Save text to file
                    
                    print(f"✅ Saved: {txt_path}")
                except Exception as e:
                    print(f"❌ Error processing {pdf_path}: {e}")

# Run the batch processing
generate_text_folder(MAIN_PDF_FOLDER, OUTPUT_FOLDER)

print("✅ All PDFs processed successfully!")