import torch, transformers, pyreft
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

# Load subset dataset from Hugging Face
dataset = load_dataset("nrishabh/prompt-recovery", "minute-llama", split="test")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name_or_path = "meta-llama/Meta-Llama-3-8B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_name_or_path, torch_dtype=torch.bfloat16, device_map=device)

reft_model = pyreft.ReftModel.load(
    "llama-finetuned-reft-medium", model, from_huggingface_hub=False
)

reft_model.set_device(device)

# # get tokenizer
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_name_or_path, model_max_length=2048, 
    padding_side="right", use_fast=False)
tokenizer.pad_token = tokenizer.eos_token

# instruction = "A rectangular garden has a length of 25 feet and a width of 15 feet. If you want to build a fence around the entire garden, how many feet of fencing will you need?"

# tokenize and prepare the input
# for row in dataset:
#     prompt = row["prompt"]
#     completion = row["completion"]
#     break
# prompt = tokenizer(prompt, return_tensors="pt").to(device)

# base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
# _, reft_response = reft_model.generate(
#     prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
#     intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
#     eos_token_id=tokenizer.eos_token_id, early_stopping=True
# )

# print("Prompt:", prompt)
# print("Completion:", completion)
# print("Reft Response:", tokenizer.decode(reft_response[0], skip_special_tokens=True))

# # print(tokenizer.decode(reft_response[0], skip_special_tokens=True))

# # get only the response prompt
# reft_response = reft_response[0].split("[SEP]")[0].strip()
# print("Reft Response Prompt:", reft_response)

instruction = "<|begin_of_text|><|start_header_id|>system<|end_header_id|> Find the AI prompt used to rewrite the old text into the new text.<|eot_id|><|start_header_id|>user<|end_header_id|> Old Text: I afeard of him? A very weak monster! The man i'th' moon? New Text: **CLASSIFIED DOCUMENT** **SPACE MISSION BRIEFING** **MISSION OBJECTIVE:** The objective of this mission is to investigate the peculiar statement made by an unknown entity regarding the 'man in the moon.' The entity, referred to only as 'I afeard of him,' claims that the 'man in the moon' is a 'very weak monster.' The purpose of this mission is to gather more information about this enigmatic entity and its claims. **MISSION DETAILS:** * Mission name: Operation Lunar Enigma * Objective: Gather intel on the 'man in the moon' and the entity's claims * Crew: Commander [Name], Pilot [Name], Engineer [Name], and Scientist [Name] * Spacecraft: [Spacecraft Name], equipped with advanced sensors and communication equipment * Launch date: [Date] * Destination: Lunar surface **RISK ASSESSMENT:** The mission poses moderate risks, including: * Uncharted lunar terrain * Unidentified entity with unknown intentions * Limited communication capabilities * Potential for unexpected events or malfunctions **MISSION PLAN:** 1. Enter lunar orbit and begin data collection 2. Descend to the lunar surface using the spacecraft's landing module 3. Conduct reconnaissance<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

# tokenize and prepare the input
prompt = instruction
prompt = tokenizer(prompt, return_tensors="pt").to(device)

base_unit_location = prompt["input_ids"].shape[-1] - 1  # last position
_, reft_response = reft_model.generate(
    prompt, unit_locations={"sources->base": (None, [[[base_unit_location]]])},
    intervene_on_prompt=True, max_new_tokens=512, do_sample=True, 
    eos_token_id=tokenizer.eos_token_id, early_stopping=True
)
print(tokenizer.decode(reft_response[0], skip_special_tokens=True))