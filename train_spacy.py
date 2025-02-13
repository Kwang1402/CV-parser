import os
import spacy
from spacy.training import Example
from spacy.util import minibatch
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json

TEXT_FOLDER = os.path.abspath("cvs_data_text")
MODEL_OUTPUT_DIR = "ner_model"
LOSS_PLOT_PATH = "training_loss.png"
OUTPUT_FILE = "model_comparison_results.txt"

# Load text files to data
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
    return [(doc["text"], {"entities": extract_entities(doc["text"])}) for doc in documents if extract_entities(doc["text"])]

training_data = prepare_training_data(documents)
if not training_data:
    print("No valid training data extracted.")
    exit()

# Split dataset into 80-20 for train-test
train_set, test_set = train_test_split(training_data, test_size=0.2, random_state=42)

# Train and save NER model
def train_ner(nlp, training_data, output_dir, n_iter=100):
    ner = nlp.add_pipe("ner") if "ner" not in nlp.pipe_names else nlp.get_pipe("ner")
    
    for _, annotations in training_data:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])
    
    losses_per_iteration = []
    
    for i in range(n_iter):
        random.shuffle(training_data)
        losses = {}
        for batch in minibatch(training_data, size=8):
            for text, annotations in batch:
                example = Example.from_dict(nlp.make_doc(text), annotations)
                nlp.update([example], drop=0.5, losses=losses)
        losses_per_iteration.append(losses.get("ner", 0))
        print(f"Iteration {i+1}: Loss {losses.get('ner', 0)}")
    
    nlp.to_disk(output_dir)
    print(f"Model saved to {output_dir}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_iter + 1), losses_per_iteration, marker='o', linestyle='-', color='b')
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("NER Training Loss per Iteration")
    plt.grid()
    plt.savefig(LOSS_PLOT_PATH)
    print(f"Loss plot saved as {LOSS_PLOT_PATH}")

train_ner(nlp, train_set, MODEL_OUTPUT_DIR)

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
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1_score": f1_score}

# Load trained model and evaluate
nlp_trained = spacy.load(MODEL_OUTPUT_DIR)
nlp_pretrained = spacy.load("en_core_web_sm")
trained_stats = evaluate_model(nlp_trained, test_set)
pretrained_stats = evaluate_model(nlp_pretrained, test_set)

# Save evaluation results
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write("=== Model Evaluation Results ===\n")
    f.write("\nTrained Model Stats:\n" + json.dumps(trained_stats, indent=4))
    f.write("\n\nPre-trained Model Stats:\n" + json.dumps(pretrained_stats, indent=4))
    f.write("\n\nSample Predictions from Trained Model (First 5 Samples):\n")
    for text, _ in test_set[:5]:
        doc = nlp_trained(text)
        f.write(f"\nText: {text[:200]}...\n")
        f.write("Entities:\n")
        for ent in doc.ents:
            f.write(f"{ent.text} - {ent.label_}\n")

print(f"Results saved to {OUTPUT_FILE}")