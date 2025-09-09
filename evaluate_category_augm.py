# evaluate_category_augm.py

from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    DataCollatorWithPadding
)
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# 1. We load splits
data_files = {
    "validation": "val.csv",
    "test":       "test.csv"
}
ds = load_dataset("csv", data_files=data_files)

# 2. Label encoding (like in training)
categories = sorted(ds["validation"].unique("category"))
ds = ds.cast_column("category", ClassLabel(names=categories))

# 3. Tokenizer & Model (we load the fine‑tuned augmented)
tokenizer = AutoTokenizer.from_pretrained("nautivoice-category-augm")
model     = AutoModelForSequenceClassification.from_pretrained("nautivoice-category-augm")

# --- Debug: make sure that they were loaded OK ---
print("Loaded model with labels:", model.config.id2label)
print("Validation split size before tokenization:", len(ds["validation"]))
print("Test split size before tokenization:      ", len(ds["test"]))

# 4. Preprocessing (tokenization + labels)
def preprocess(batch):
    tokens = tokenizer(
        batch["report"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    tokens["labels"] = batch["category"]
    return tokens

val_dataset = ds["validation"].map(
    preprocess,
    batched=True,
    remove_columns=ds["validation"].column_names
)
test_dataset = ds["test"].map(
    preprocess,
    batched=True,
    remove_columns=ds["test"].column_names
)

# --- Debug: make sure that there is data after map ---
print("Validation split size after tokenization:", len(val_dataset))
print("Test split size after tokenization:      ", len(test_dataset))

# 5. Data collator
data_collator = DataCollatorWithPadding(tokenizer)

# 6. Trainer only for evaluate/predict
trainer = Trainer(
    model=model,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# 7. Evaluate in validation
metrics_val = trainer.evaluate(val_dataset)
print("\n=== Validation metrics ===")
print(metrics_val)

# 8. Predict + sklearn report in test
preds_output = trainer.predict(test_dataset)
preds  = np.argmax(preds_output.predictions, axis=-1)
labels = preds_output.label_ids

print("\n=== Test Accuracy & F1 ===")
print("Accuracy:   ", accuracy_score(labels, preds))
print("F1 weighted:", f1_score(labels, preds, average="weighted"))

print("\n=== Classification Report ===")
print(classification_report(labels, preds, target_names=categories))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(labels, preds))
