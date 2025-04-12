import argparse
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

__all__ = []


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract the last token of the last layer of the transformer."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def main(args) -> None:
    """Main function."""

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    model = model.eval().cuda()
    max_length = 4096

    # list all .parquet files in the directory
    parquet_files = list(Path(args.cc12m_root).rglob("*.parquet"))
    parquet_files = [str(p) for p in parquet_files]

    # split them
    if args.index is not None:
        parquet_files = np.array_split(parquet_files, 64)[args.index]
        print(
            f"Processing {len(parquet_files)} parquet files starting from {parquet_files[0]}"
        )  # noqa: T201

    if args.reverse:
        parquet_files = parquet_files[::-1]

    print(f"Found {len(parquet_files)} parquet files")  # noqa: T201

    output_dir = os.path.join(args.output_dir, "embeddings")
    os.makedirs(output_dir, exist_ok=True)

    for p_file in tqdm(parquet_files):
        start_time = time.time()
        filename = os.path.basename(p_file).split(".")[0]
        output_file = os.path.join(output_dir, f"{filename}.npy")

        if os.path.exists(output_file):
            print(f"Skipping {filename}")  # noqa: T201
            continue

        df = pd.read_parquet(p_file)

        file_embeddings = []

        # for chunk in tqdm(df, leave=False):
        captions = df["caption"].values.tolist()

        for chunk_start in range(0, len(captions), 128):
            captions_chunk = captions[chunk_start : chunk_start + 128]  # noqa: E203

            batch_dict = tokenizer(
                captions_chunk,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
            file_embeddings.extend(embeddings)

        file_embeddings = np.array(file_embeddings)
        np.save(output_file, file_embeddings)

        end_time = time.time()
        print(f"Processed {filename} in {end_time - start_time:.2f} seconds")  # noqa: T201


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--c12m_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--reverse",
        action="store_true",
        description="Reverse the order of the files, useful for running on multiple GPUs as the second GPU will process the last files first.",
    )
    parser.add_argument("--index", type=int)

    args = parser.parse_args()
    main(args)
