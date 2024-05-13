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
dataset = load_dataset("nrishabh/prompt-recovery", "minute", split="test")

print(dataset[0])

# Test on a small subset of two data points
# dataset = dataset.select(range(100))

# Prepare input text by concatenating 'original_text_text' and 'rewritten_text'
prompt_text = "Generate only the prompt no more than 8 words used to convert the Original Text into Rewritten Text:"
inputs = [f"{item['original_text']} [SEP] {prompt_text} [SEP] {item['rewritten_text']}" for item in dataset]
targets = [item['prompt'] for item in dataset]


# Load tokenizer and model for Meta-Llama-3-8B-Instruct
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda" if torch.cuda.is_available() else "cpu",
)

# Process and generate outputs
outputs = []
uni_bleu_scores = []
bi_bleu_scores = []
tri_bleu_scores = []


uni_bleu = BLEUScore(n_gram=1)
bi_bleu = BLEUScore(n_gram=2)
tri_bleu = BLEUScore(n_gram=3)


for i, (input_text, target_text) in enumerate(zip(inputs, targets)):
    # Generate the prompt using the chat template
    prompt = pipeline.tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a rewriting assistant!"},
         {"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True
    )

    # Define terminators for the end of generation
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    # Generate output sequences
    output_sequences = pipeline(
        prompt,
        max_new_tokens=500,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    # Extract and decode the generated text
    output_text = output_sequences[0]["generated_text"][len(prompt):].strip()
    outputs.append(output_text)

    # Calculate BLEU score
    uni_bleu_score = uni_bleu([output_text], [[target_text]])
    bi_bleu_score = bi_bleu([output_text], [[target_text]])
    tri_bleu_score = tri_bleu([output_text], [[target_text]])

    uni_bleu_scores.append(uni_bleu_score.item())
    bi_bleu_scores.append(bi_bleu_score.item())
    tri_bleu_scores.append(tri_bleu_score.item())

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

hyp, ref = outputs, targets
rouge = Rouge()
scores = rouge.get_scores(hyp, ref, avg=True)
for key, value in scores.items():
    print(f"{key}: {value}")
    logging.info(f"{key}: {value}")