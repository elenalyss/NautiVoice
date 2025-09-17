import pandas as pd
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Loading of validation data
df = pd.read_csv("val.csv")
df = df.dropna(subset=["report", "severity"])

# 2. Severity labels (they need to be in the same order like in the training)
labels = ["Low", "Medium", "High", "Critical"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# 3. Model loading and tokenizer
model_path = "nautivoice-severity-augm-2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# 4. Predictions
y_true = []
y_pred = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row["report"]
    true_label = row["severity"]
   
    # Tokenization
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
   
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
   
    y_true.append(label2id[true_label])
    y_pred.append(pred_id)

# 5. Evaluation
print("Accuracy:", accuracy_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred, average="weighted"))
print("Classification Report Augm 2:")
print(classification_report(y_true, y_pred, target_names=labels))

#confusion matrix
cm=confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Severity Classification - Augm 2')
plt.tight_layout()
plt.show()