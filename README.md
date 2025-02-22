# CV-Parser

A NER model that extracts information from CVs.

## Requirements

-   [Python 3.11.0](https://www.python.org/downloads/release/python-3110/)

# Getting started

## Installation

-   Clone the repository
-   Install the dependencies:

    ```
    pip install -r requirements.txt
    ```

-   Install spaCy's `en_core_web_sm` model:

    ```
    python -m spacy download en_core_web_sm
    ```

# Data preprocessing

-   `preprocess.py` contains ResumeProcessor class

    In this file, you need to download these:

    ```
    nltk.download("stopwords") 
    nltk.download("wordnet")
    ```
    By running:

    ```
    python preprocess.py
    ```
-   `pdf_to_text.py` is the main file using ResumeProcessor class to convert PDF data in `cvs_data` folder to text and write it to the `cvs_data_text` folder. Specify `tesseract_path` in `config.json` then run:

    ```
    python pdf_to_text.py
    ```


# Training the Model

-   To train the model, run `train_spacy.py`:

    ```bash
    python train_spacy.py
    ```
    The trained model can be found at the `ner_model` folder

# Evaluating the Model

-   The trained model is evaluated against spaCy's `en_core_web_sm` model. Run `test.py`:

    ```
    python test.py
    ```
-   The results can be found at `results.txt`
