from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# 1. Load csv as dataset
data_files = {"train": "train_augmented.csv", "validation": "val.csv"}
ds = load_dataset("csv", data_files=data_files)

# 2. Category convert into arithmetic labels
categories = sorted(ds["train"].unique("category"))
ds = ds.cast_column("category", ClassLabel(names=categories))

# 3. Tokenizer & Model
model_name = "distilbert-base-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(categories)
)

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

tokenized = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds["train"].column_names
)

# 5. TrainingArguments (simple setup)
training_args = TrainingArguments(
    output_dir="cat_model",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="cat_model/logs",
    logging_steps=100
)

# 6. compute_metrics for accuracy & F1
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }

# 7. Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer)

# 8. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# 9. Fine‑tuning
trainer.train()

# 10. final model save
trainer.save_model("nautivoice-category-augm")
