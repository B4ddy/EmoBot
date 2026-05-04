from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import load_dataset

# Parameters as in the autotrainer
model_id = "google-bert/bert-base-uncased"

# Load dataset (using 'emotion' as an example with 6 labels matching num_labels=6)
ds = load_dataset("dair-ai/emotion") # for local dataset: load_dataset("csv", data_files={"train": "train.csv", "validation": "validation.csv"}, delimiter=",")

# Load tokenizer and tokenize the dataset
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

tokenized_ds = ds.map(tokenize, batched=True)

# Parameters as in the autotrainer
training_args = TrainingArguments(
    output_dir="models/emotion_classifier",  # Output directory for checkpoints and final model
    per_device_train_batch_size=8,  #  Batch size
    learning_rate=2e-5,             # Learning rate (0.00002)
    num_train_epochs=3,             #  Epochs
    bf16=True,                      #  (bf16 for Blackwell/sm_120) rtx 5000 series, fp16 for others
    eval_strategy="epoch",
    save_strategy="epoch",          # Only save at end of each epoch (not every x steps)
    save_total_limit=2,             
    load_best_model_at_end=True,    # Load the best checkpoint when training finishes
)

# Parameters as in the autotrainer
trainer = Trainer(
    model=AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=6),
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"]
)

trainer.train()

# Save the final model + tokenizer together so the output dir is self-contained
trainer.save_model("emotion_classifier")
tokenizer.save_pretrained("emotion_classifier")
