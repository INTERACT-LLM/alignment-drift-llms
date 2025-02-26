"""
Compute perplexity

Code modified from: https://huggingface.co/docs/transformers/perplexity (mainly updated)
"""
from pathlib import Path

from utils import read_data
import polars as pl

from evaluate import load 

def compute_perplexity(texts:list, model_id:str = "gpt2", batch_size:int = 1, max_length:int = 1024):
    '''
    Compute perplexity 

    This perplexity "is defined as the exponentiated average negative log-likelihood of a sequence, calculated with exponent base `e`."
    source: https://huggingface.co/spaces/evaluate-measurement/perplexity/blob/main/README.md

    Args:
        texts: list of texts
        model_id: model id 
        batch_size: batch size for processing
    '''
    perplexity = load("perplexity", module_type="metric")
    
    
    ppl_scores = perplexity.compute(predictions=texts, 
                                            model_id=model_id, 
                                            add_start_token=True, # (default to be able to compute perplexity of first token see: https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py)
                                            batch_size=batch_size,
                                            max_length=max_length,
                                            )
    return ppl_scores

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

    # compute perplexity
    model_id = "gpt2"
    texts = df["content"].to_list()
    ppl = compute_perplexity(texts, model_id, batch_size=10)

    # add to df
    df = df.with_columns(perplexity=pl.Series(ppl["perplexities"]))

    df.write_csv(data_path.parents[1] / f"v{version}_perplexity.csv")

    # plot (filter first)
    role = "assistant"
    df = df.filter(role = role)
    df = df.with_columns(total_message_number=pl.int_range(1, pl.len() + 1).over("id"))
    avg_df = df.group_by(["group", "total_message_number"], maintain_order=True).mean()

    plot = (avg_df.plot.line(
                x="total_message_number", 
                y="perplexity", 
                color="group"
                )
                .properties(width=600, height=600, title=f"Average Perplexity Score Per Group (n = 6)")
                .configure_scale(zero=False)
        )

    plot.encoding.y.title = "Average Perplexity (GPT-2c computed)"
    plot.encoding.x.title = f"Message number ({role} only)"


    save_dir = Path(__file__).parents[1] / "plots"
    save_dir.mkdir(parents=True, exist_ok=True)
    plot.save(save_dir / f"v{version}_perplexity_plot.html")


if __name__ == "__main__":
    main()