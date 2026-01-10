import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "❌ Le dossier 'model/' est introuvable. "
        "Veuillez d'abord entraîner le modèle avec train_bert.py"
    )

tokenizer = DistilBertTokenizerFast.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)

model.eval()

def predict_toxicity(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    toxic_score = probs[0][1].item() * 100

    return toxic_score
