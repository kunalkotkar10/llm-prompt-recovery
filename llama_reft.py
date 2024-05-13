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

dataset_size = 'minute'

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

# Load subset dataset from Hugging Face
dataset = load_dataset("nrishabh/prompt-recovery", f"{dataset_size}-llama", split="train")
val_dataset = load_dataset("nrishabh/prompt-recovery", f"{dataset_size}-llama", split="validation")

print(dataset[0])

# print dataset
# for i, row in enumerate(dataset):
#     print(f"Prompt:", row["prompt"])
#     print(f"Completion:", row["completion"])

#     if i > 5:
#         break

device = "cuda" if torch.cuda.is_available() else "cpu"

# prompt_no_input_template = """<|begin_of_text|><|start_header_id|>user<|end_header_id|>%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

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

# supervised ft
data_module = pyreft.make_last_position_supervised_data_module(
    tokenizer, model, [row["prompt"] for row in dataset], 
    [row["completion"] for row in dataset])

# print(data_module[0])

# train
training_args = transformers.TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 8,
    warmup_steps = 100,
    num_train_epochs = 50.0,
    learning_rate = 4e-3,
    bf16 = True,
    logging_steps = 1,
    optim = "paged_adamw_32bit",
    weight_decay = 0.0,
    lr_scheduler_type = "cosine",
    output_dir = "outputs",
    report_to=[]
)

trainer = pyreft.ReftTrainerForCausalLM(model=reft_model, 
            tokenizer=tokenizer, args=training_args, **data_module)

print("Training model")
_ = trainer.train()

# save model
reft_model.save(
    save_directory=f"./llama-finetuned-reft_{dataset_size}", 
)

# trainer.push_to_hub(
#     model_name="trained_reft", 
#     repo_url="kkotkar1/prompt-recovery", 
#     commit_message="Add reft model trained on prompt recovery dataset"
# )

reft_model.save(
    save_directory=f"./llama-finetuned-reft_{dataset_size}", 
    save_to_hf_hub=True, 
    hf_repo_name="kkotkar1/llama3-reft"
)

# upload model to huggingface
# model_push = push_to_hub(
#     model, 
#     model_name="trained_reft", 
#     repo_url="kkotkar1/prompt-recovery", 
#     use_temp_dir=True, 
#     commit_message="Add reft model trained on prompt recovery dataset"
# )