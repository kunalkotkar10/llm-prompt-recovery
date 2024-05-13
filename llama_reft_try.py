import torch
import transformers
import wandb
import logging
import json
import os
from dotenv import load_dotenv
from huggingface_hub import login
from torchmetrics.text import BLEUScore
from rouge import Rouge
import pyreft

# Load environment variables
load_dotenv()

huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token:
    print("Hugging Face token loaded successfully.")
else:
    print("Error: Hugging Face token not found. Please check your .env file.")

# Log in to Hugging Face
login(token=huggingface_token)
print("Logged in to Hugging Face successfully.")

# import dataset from huggingface datasets
from datasets import load_dataset

# Load train, validation, and test datasets
train_dataset = load_dataset("nrishabh/prompt-recovery", "minute-llama", split="train")
val_dataset = load_dataset("nrishabh/prompt-recovery", "minute-llama", split="validation")
test_dataset = load_dataset("nrishabh/prompt-recovery", "minute-llama", split="test")


print(train_dataset[0])

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model")
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device, trust_remote_code=True)

# get tokenizer
print("Loading tokenizer")
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# get reft model
print("Loading reft model")
reft_config = pyreft.ReftConfig(representations={
    "layer": 8, "component": "block_output",
    "low_rank_dimension": 4,
    "intervention": pyreft.LoreftIntervention(embed_dim=model.config.hidden_size,
    low_rank_dimension=4)})
reft_model = pyreft.get_reft_model(model, reft_config)
reft_model.set_device(device)
reft_model.print_trainable_parameters()

# Prepare data modules
train_data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [row["prompt"] for row in train_dataset], 
    [row["completion"] for row in train_dataset])

val_data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [row["prompt"] for row in val_dataset], 
    [row["completion"] for row in val_dataset])

# Train
training_args = transformers.TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    warmup_steps = 100,
    num_train_epochs = 3,
    learning_rate = 5e-4,
    bf16 = True,
    logging_steps = 1,
    optim = "paged_adamw_32bit",
    weight_decay = 0.0,
    lr_scheduler_type = "cosine",
    output_dir = "outputs",
    evaluation_strategy="steps",
    eval_steps=50,
    report_to=[]
)

trainer = pyreft.ReftTrainerForCausalLM(
    model=reft_model, 
    tokenizer=tokenizer, 
    args=training_args, 
    **train_data_module,
    eval_dataset=val_data_module
)

print("Training model")
_ = trainer.train()

# Save the trained model
reft_model.save(save_directory="./llama-finetuned-reft-try")

trainer.push_to_hub(
    model_name="trained_reft", 
    repo_url="kkotkar1/prompt-recovery", 
    commit_message="Add reft model trained on prompt recovery dataset"
)

# Evaluate on the test dataset
print("Evaluating model on test dataset")

# Generate predictions on the test set
test_prompts = [row["prompt"] for row in test_dataset]
test_references = [row["completion"] for row in test_dataset]
test_predictions = []

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = reft_model.generate(**inputs)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    test_predictions.append(decoded_output)

# Calculate ROUGE score
rouge = Rouge()
scores = rouge.get_scores(test_predictions, test_references, avg=True)

print("ROUGE scores:", json.dumps(scores, indent=2))
