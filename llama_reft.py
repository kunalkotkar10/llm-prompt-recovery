from huggingface_hub import notebook_login
notebook_login()

dataset_name = "teknium/OpenHermes-2.5"
from datasets import load_dataset

dataset = load_dataset(dataset_name, split="train")
dataset = dataset.select(range(100))

# data_module = pyreft.make_last_position_supervised_data_module(
#     tokenizer, model, [prompt_no_input_template % row["conversations"][0]["value"] for row in dataset], 
#     [row["conversations"][1]["value"] for row in dataset])

print(dataset[0])