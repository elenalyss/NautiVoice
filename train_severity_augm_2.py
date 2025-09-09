
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
import torch
from torch.nn import CrossEntropyLoss
 
# 1. load of CSV
data_files = {"train": "train_augmented.csv", "validation": "val.csv"}
ds = load_dataset("csv", data_files=data_files)
 
# 2. Severity labels
severity_labels = ["Low", "Medium", "High", "Critical"]
ds = ds.cast_column("severity", ClassLabel(names=severity_labels))
 
# 3. Tokenizer & Model
model_name = "distilbert-base-cased"
tokenizer  = AutoTokenizer.from_pretrained(model_name)
model      = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(severity_labels)
)
 
# weights calculation per class
counts = np.bincount(ds["train"]["severity"])
total  = counts.sum()
class_weights = torch.tensor([ total / c for c in counts ], dtype=torch.float)
 
# 4. Preprocess
def preprocess(batch):
    toks = tokenizer(
        batch["report"],
        truncation=True,
        padding="max_length",
        max_length=128
    )
    toks["labels"] = batch["severity"]
    return toks
 
tokenized = ds.map(
    preprocess,
    batched=True,
    remove_columns=ds["train"].column_names
)
 
# 5. TrainingArguments
training_args = TrainingArguments(
    output_dir="sev_model_augm_2",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    save_total_limit=2,
    logging_dir="sev_model_augm_2/logs",
    logging_steps=100
)
 
# 6. Subclass Trainer for weighted loss
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        loss_fct = CrossEntropyLoss(weight=class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
 
# 7. Trainer setup
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=-1)),
        "f1":       f1_score(p.label_ids, np.argmax(p.predictions, axis=-1), average="weighted"),
    }
)
 
# 8. Fine-tuning & save
trainer.train()
trainer.save_model("nautivoice-severity-augm-2")