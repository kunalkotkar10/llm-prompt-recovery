import torch
import wandb
import argparse
import transformers
import pandas as pd
from tqdm.auto import tqdm
from datasets import load_dataset
from torchmetrics.text import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from dotenv import load_dotenv

def prepare_dataset(ds: list[dict]) -> list[list[dict]]:

    return [
        [
            {
                "role": "system",
                "content": "Find the AI prompt used to rewrite the old text into the new text.",
            },
            {
                "role": "user",
                "content": f"Old Text: {ds[i]['original_text']}\nNew Text: {ds[i]['rewritten_text']}",
            },
        ]
        for i in range(len(ds))
    ]


def calc_metrics(preds: list, targets: list, run):
    uni_bleu = BLEUScore(n_gram=1)
    bi_bleu = BLEUScore(n_gram=2)
    tri_bleu = BLEUScore(n_gram=3)
    rouge = ROUGEScore()

    bleu_unigram = uni_bleu(preds, targets)
    bleu_bigram = bi_bleu(preds, targets)
    bleu_trigram = tri_bleu(preds, targets)
    rouge_score = rouge(preds, targets)

    run.log(
        {
            "bleu/unigram": bleu_unigram,
            "bleu/bigram": bleu_bigram,
            "bleu/trigram": bleu_trigram,
            "rouge/rouge1/precision": rouge_score["rouge1_precision"],
            "rouge/rouge1/fmeasure": rouge_score["rouge1_fmeasure"],
            "rouge/rouge1/recall": rouge_score["rouge1_recall"],
            "rouge/rouge2/precision": rouge_score["rouge2_precision"],
            "rouge/rouge2/fmeasure": rouge_score["rouge2_fmeasure"],
            "rouge/rouge2/recall": rouge_score["rouge2_recall"],
            "rouge/rougeL/precision": rouge_score["rougeL_precision"],
            "rouge/rougeL/fmeasure": rouge_score["rougeL_fmeasure"],
            "rouge/rougeL/recall": rouge_score["rougeL_recall"],
            "rouge/rougeLsum/precision": rouge_score["rougeLsum_precision"],
            "rouge/rougeLsum/fmeasure": rouge_score["rougeLsum_fmeasure"],
            "rouge/rougeLsum/recall": rouge_score["rougeLsum_recall"],
        }
    )


def run_inference(
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
    ds: list[list[dict]],
):
    """
    Run inference on a list of inputs using a tokenizer and model
    """
    messages = prepare_dataset(ds)
    outputs = list()
    for message in tqdm(messages, desc="processing inputs", leave=False):

        input_ids = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]

        output = model.generate(
            input_ids,
            max_new_tokens=128,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        outputs.append(
            tokenizer.decode(output[0][input_ids.shape[-1] :], skip_special_tokens=True)
        )

    return outputs

def main(dataset_name="nrishabh/prompt-recovery", dataset_subset="mini", model_names=["nrishabh/llama3-8b-instruct-qlora-mini"]):

    dataset = load_dataset(dataset_name, dataset_subset, split="test")

    for model_id in tqdm(model_names, desc="running inference"):
        run = wandb.init(
            entity="jhu-llm-prompt-recovery",
            project="llm-prompt-recovery",
            job_type="inference",
            name=f"{model_id}",
        )
        ## TODO YOUR THING
        # outputs = run_inference(tokenizer, model, dataset)
        test_outputs = [
            {
                "original_text": dataset["original_text"][i],
                "rewritten_text": dataset["rewritten_text"][i],
                "expected_prompt": dataset["prompt"][i],
                "predicted_prompt": outputs[i],
            }
            for i in range(len(outputs))
        ]
        run.log({"tests_outputs": wandb.Table(data=pd.DataFrame(test_outputs))})
        targets = [[dataset["prompt"][i]] for i in range(len(dataset["prompt"]))]
        calc_metrics(outputs, targets, run)
        run.finish()