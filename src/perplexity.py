"""
Compute perplexity

Code modified from: https://huggingface.co/docs/transformers/perplexity (mainly updated)
"""
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm

from utils import read_data

def compute_perplexity(encodings, seq_len:int, stride:int, model: AutoModelForCausalLM, device="mps"):
    # compute max length
    max_length = model.config.n_positions

    # perplexity 
    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100     

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)

    return ppl

def main():
    # read data 
    version = 1.1

    data_path = (
        Path(__file__).parents[1]
        / "data"
        / "mlx-community--Qwen2.5-7B-Instruct-1M-4bit"
        / f"v{version}"
    )   

    df = read_data(data_dir=data_path)

    device = "mps"

    # load mdl 
    model_id = "openai-community/gpt2-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

    # compute 
    perplexity_scores = []

    for row in df.iter_rows(named=True):
        text = row["content"]

        encodings = tokenizer(text, return_tensors="pt")
        seq_len = encodings.input_ids.size(1)
        stride = 100

        ppl = compute_perplexity(encodings, seq_len, stride, model)

        perplexity_scores.append(ppl)
    
    print(perplexity_scores)


if __name__ == "__main__":
    main()