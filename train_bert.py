import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

# =========================
# PARAMÈTRES D'ENTRAÎNEMENT
# =========================
MODEL_NAME = "distilbert-base-uncased"
MODEL_SAVE_PATH = "model"

EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
MAX_LENGTH = 128

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# CHARGEMENT DU DATASET
# =========================
df = pd.read_csv("data/dataset.csv")

X_train, X_val, y_train, y_val = train_test_split(
    df["clean_tweet"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

# =========================
# TOKENIZER & DATASET
# =========================
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)

class TweetDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH
        )
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = TweetDataset(X_train, y_train)
val_dataset = TweetDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# =========================
# MODÈLE
# =========================
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# =========================
# ENTRAÎNEMENT
# =========================
print("🚀 Début de l'entraînement...\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for batch in train_loader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # =========================
    # VALIDATION
    # =========================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_accuracy = correct / total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f}"
    )

# =========================
# SAUVEGARDE DU MODÈLE
# =========================
model.save_pretrained(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)

print("\n✅ Modèle entraîné et sauvegardé dans le dossier 'model/'")
