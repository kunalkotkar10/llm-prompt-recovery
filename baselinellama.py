import transformers
import wandb
import logging
import json
from dotenv import load_dotenv

load_dotenv()

# Set up logging configuration
logging.basicConfig(filename='model_outputs.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load data from the downloaded artifact
artifact_dir = 'arxiv-abstracts-dataset_20240420_031345:v0'
json_name = 'arxiv-abstracts-dataset_20240420_031345.json'
with open(f'artifacts/{artifact_dir}/{json_name}', 'r') as file:
    data = json.load(file)
print('Data loaded successfully', data[0])

prompt_text = "Generate the prompt used to rewrite the above text into the following text:"

# Prepare input text by concatenating 'original_text_text' and 'rewritten_text'
inputs = [f"{item['original_text_text']} [SEP] {prompt_text} [SEP] {item['rewritten_text']}" for item in data]

targets = [item['instruction_text'] for item in data]

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
for i, input_text in enumerate(inputs):
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
        pipeline.tokenizer.convert_tokens_to_ids("")
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
    output_text = output_sequences[0]["generated_text"][len(prompt):]
    outputs.append(output_text)
    print('Output generated successfully')
    # if i > 5:
    #     break

print('Outputs generated successfully')

# Log results, optionally compare with targets
for output, target in zip(outputs, targets):
    logging.info(f"Generated: {output}")
    logging.info(f"Expected: {target}")