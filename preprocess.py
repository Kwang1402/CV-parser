import re
import cv2
import numpy as np
import pytesseract
import nltk
from pdf2image import convert_from_path
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

'''just need download once'''
# nltk.download("stopwords")
# nltk.download("wordnet")

class ResumeProcessor:
    def __init__(self, tesseract_path=None):
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def ocr_image(self, image):
        image_gray = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
        image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]  # Improve OCR accuracy
        text = pytesseract.image_to_string(image_gray)
        return text

    def extract_text_from_pdf(self, pdf_path):
        images = convert_from_path(pdf_path) 
        extracted_text = []

        for image in images:
            text = self.ocr_image(image)
            extracted_text.append(text)

        return "\n".join(extracted_text).strip()

    def clean_text(self, text):
        text = re.sub(r"(@[A-Za-z0-9_]+)|([^0-9A-Za-z \t])|(http\S+)", " ", text)
        words = text.split()
        words = [self.lemmatizer.lemmatize(word) for word in words if word.lower() not in self.stop_words]
        cleaned_text = " ".join(words)
        return cleaned_text

    def process_resume(self, pdf_path):
        extracted_text = self.extract_text_from_pdf(pdf_path)
        cleaned_text = self.clean_text(extracted_text)
        return cleaned_text

