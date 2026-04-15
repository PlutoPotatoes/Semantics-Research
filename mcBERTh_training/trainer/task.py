from transformers import (
    BertForMaskedLM, AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from data_streamer import build_decade_balanced_stream, DECADES
import os
import json
import math
from google.cloud import storage
from google.oauth2 import service_account
import math

# ── Hyperparameters ───────────────────────────────────────────────
model_name = "emanjavacas/MacBERTh"
experiment_name = "McBERTh-Pretrain-v1"
epochs = 3
learning_rate = 5e-5
batch_size = 32
# gradient_accumulation_steps = (batchsize * 8) / (batchsize * #_GPU)
# 1 GPU:  256 / (32 * 1) = 8
# 2 GPUs: 256 / (32 * 2) = 4
gradient_accumulation_steps = 8
max_steps = math.ceil(
    2102849 / (batch_size * gradient_accumulation_steps)) * epochs
logging_steps = 100
warmup_ratio = 0.05
weight_decay = 0.01
save_steps = 500
mlm_probability = 0.15
gcs_credentials = "nlp-research-sp26-8499634f1c62.json"

# ── Tokenizer ─────────────────────────────────────────────────────


def get_date_tokens(decades):
    return [f"<decade_{str(d).removesuffix('s')}>" for d in decades]


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens(
    {'additional_special_tokens': get_date_tokens(DECADES)})


def tokenize_data(examples):
    result = tokenizer(examples["text"], max_length=512, truncation=True)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(
            i) for i in range(len(result["input_ids"]))]
    return result


# ── Datasets ──────────────────────────────────────────────────────
train_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials)
val_dataset = build_decade_balanced_stream(
    service_account_path=gcs_credentials, split='valid')
train_dataset = train_dataset.map(
    tokenize_data, batch_size=batch_size, batched=True)
val_dataset = val_dataset.map(
    tokenize_data,   batch_size=batch_size, batched=True)

# ── Model ─────────────────────────────────────────────────────────
model = BertForMaskedLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=mlm_probability
)

# ── Training ──────────────────────────────────────────────────────
# Vertex AI sets AIP_MODEL_DIR automatically — use it as output dir
output_dir = os.environ.get("AIP_MODEL_DIR", f"Models/{experiment_name}")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    max_steps=max_steps,
    eval_strategy='steps',
    eval_steps=save_steps,
    save_strategy='steps',
    save_steps=save_steps,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    logging_steps=logging_steps,
    warmup_ratio=warmup_ratio,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    optim='adamw_torch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(
        early_stopping_patience=3, early_stopping_threshold=0.001)]
)

trainer.train()

# ── Save ──────────────────────────────────────────────────────────
save_path = f"{output_dir}/best"
trainer.save_model(save_path)
tokenizer.save_pretrained(save_path)

# Save parameters JSON to GCS
train_loss = [e['loss'] for e in trainer.state.log_history if 'loss' in e]
eval_loss = [e['eval_loss']
             for e in trainer.state.log_history if 'eval_loss' in e]

params = {
    "model_name": model_name,
    "max_steps": max_steps,
    "batch_size": batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "learning_rate": learning_rate,
    "warmup_ratio": warmup_ratio,
    "weight_decay": weight_decay,
    "training_loss": train_loss,
    "eval_loss": eval_loss,
}

credentials = service_account.Credentials.from_service_account_file(
    gcs_credentials)
client = storage.Client(credentials=credentials)
bucket = client.bucket("project3102-model-bucket")
blob = bucket.blob(f"Training-Tests/{experiment_name}/parameters.json")
blob.upload_from_string(json.dumps(params, indent=4),
                        content_type="application/json")
print("Done. Parameters saved to GCS.")
