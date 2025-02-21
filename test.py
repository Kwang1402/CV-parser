import os
import spacy
from sklearn.model_selection import train_test_split
import json

TEXT_FOLDER = os.path.abspath("cvs_data_text")
MODEL_DIR = "ner_model"
OUTPUT_FILE = "results.txt"


def load_text_files(text_folder):
    if not os.path.exists(text_folder):
        print(f"Error: Folder '{text_folder}' not found!")
        return []

    data = []
    for root, _, files in os.walk(text_folder):
        for filename in files:
            if filename.lower().endswith(".txt"):
                with open(os.path.join(root, filename), "r", encoding="utf-8") as f:
                    data.append({"filename": filename, "text": f.read()})
    return data


# Load data from txt files
documents = load_text_files(TEXT_FOLDER)
if not documents:
    print(f"No text files found in '{TEXT_FOLDER}'.")
    exit()
print(f"Loaded {len(documents)} resumes.")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("Loaded base spaCy model.")
except:
    print("Error loading spaCy model.")
    exit()


# Extract named entities from text
def extract_entities(text):
    return [(ent.start_char, ent.end_char, ent.label_) for ent in nlp(text).ents]


# Prepare dataset in spaCy format
def prepare_training_data(documents):
    return [
        (doc["text"], {"entities": extract_entities(doc["text"])})
        for doc in documents
        if extract_entities(doc["text"])
    ]


training_data = prepare_training_data(documents)
if not training_data:
    print("No valid training data extracted.")
    exit()

# Split dataset into 80-20 for train-test
train_set, test_set = train_test_split(training_data, test_size=0.2, random_state=42)


# Evaluate model
def evaluate_model(nlp_model, test_data):
    true_entities, pred_entities = [], []

    for text, annotations in test_data:
        doc = nlp_model(text)
        true_ents = {(ent[0], ent[1], ent[2]) for ent in annotations["entities"]}
        pred_ents = {(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents}

        true_entities.extend(true_ents)
        pred_entities.extend(pred_ents)

    true_pos = len(set(true_entities) & set(pred_entities))
    false_pos = len(set(pred_entities) - set(true_entities))
    false_neg = len(set(true_entities) - set(pred_entities))

    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    f1_score = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return {"precision": precision, "recall": recall, "f1_score": f1_score}


nlp_trained = spacy.load(MODEL_DIR)
trained_stats = evaluate_model(nlp_trained, test_set)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Model Evaluation Results ===\n")
    f.write("\nTrained Model Stats:\n" + json.dumps(trained_stats, indent=4))
    f.write("\n\nPredictions from Trained Model:\n")
    for text, _ in test_set:
        doc = nlp_trained(text)
        f.write(f"\nText: {text}...\n")
        f.write("Entities:\n")
        for ent in doc.ents:
            f.write(f"{ent.text} - {ent.label_}\n")

print(f"Results saved to {OUTPUT_FILE}")
