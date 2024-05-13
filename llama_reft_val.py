import torch
import transformers
import logging
import os
from dotenv import load_dotenv
from torchmetrics.text import BLEUScore
from rouge import Rouge
import pyreft

# Load environment variables
load_dotenv()

dataset_size='medium'

huggingface_token = os.getenv('HUGGINGFACE_TOKEN')
if huggingface_token:
    print("Hugging Face token loaded successfully.")
else:
    print("Error: Hugging Face token not found. Please check your .env file.")

# Log in to Hugging Face
from huggingface_hub import login
login(token=huggingface_token)
print("Logged in to Hugging Face successfully.")

# Load subset dataset from Hugging Face
from datasets import load_dataset
test_dataset = load_dataset("nrishabh/prompt-recovery", f"{dataset_size}-llama", split="test")

print(test_dataset[0])

# Define device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model and tokenizer
model_name_or_path = "meta-llama/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

reft_model_name = f"llama-finetuned-reft_{dataset_size}"

reft_model = pyreft.ReftModel.load(
    "kkotkar1/llama3-reft", reft_model_name, from_huggingface_hub=True
)

reft_model.set_device(device)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# Define BLEU score metrics
uni_bleu = BLEUScore(n_gram=1)
bi_bleu = BLEUScore(n_gram=2)
tri_bleu = BLEUScore(n_gram=3)

# Initialize lists for BLEU scores
uni_bleu_scores = []
bi_bleu_scores = []
tri_bleu_scores = []

# Initialize ROUGE
rouge = Rouge()

# Generate outputs and calculate metrics
outputs = []
targets = []

for i, row in enumerate(test_dataset):
    prompt = row["prompt"]
    completion = row["completion"]
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=50, do_sample=True)
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # convert tensor to string
    completion = completion[0]
    output_text = output_text[0]
    
    # Append for metric calculation
    outputs.append(output_text)
    targets.append(completion)
    
    # Calculate BLEU scores
    uni_bleu_score = uni_bleu([output_text], [[completion]])
    bi_bleu_score = bi_bleu([output_text], [[completion]])
    tri_bleu_score = tri_bleu([output_text], [[completion]])
    
    uni_bleu_scores.append(uni_bleu_score.item())
    bi_bleu_scores.append(bi_bleu_score.item())
    tri_bleu_scores.append(tri_bleu_score.item())
    
    if i < 5:  # Print first 5 examples
        print(f"Prompt: {prompt}")
        print(f"Target: {completion}")
        print(f"Output: {output_text}")
        print("-" * 50)

# Average BLEU scores
average_uni_bleu = sum(uni_bleu_scores) / len(uni_bleu_scores)
average_bi_bleu = sum(bi_bleu_scores) / len(bi_bleu_scores)
average_tri_bleu = sum(tri_bleu_scores) / len(tri_bleu_scores)

print(f"Average Uni-gram BLEU score: {average_uni_bleu}")
print(f"Average Bi-gram BLEU score: {average_bi_bleu}")
print(f"Average Tri-gram BLEU score: {average_tri_bleu}")
logging.info(f"Average Uni-gram BLEU score: {average_uni_bleu}")
logging.info(f"Average Bi-gram BLEU score: {average_bi_bleu}")
logging.info(f"Average Tri-gram BLEU score: {average_tri_bleu}")

# Calculate ROUGE scores
scores = rouge.get_scores(outputs, targets, avg=True)
for key, value in scores.items():
    print(f"{key}: {value}")
    logging.info(f"{key}: {value}")
