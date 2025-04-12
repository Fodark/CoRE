import argparse
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch
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
    model = AutoModel.from_pretrained(
        "Salesforce/SFR-Embedding-Mistral",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = model.eval().cuda()
    max_length = 4096

    # list all .parquet files in the directory
    parquet_files = list(Path(args.coyo_root).rglob("*.arrow"))
    parquet_files = [str(p) for p in parquet_files]

    # split them
    if args.index is not None:
        parquet_files = np.array_split(parquet_files, args.n_indices)[args.index]
        print(
            f"Processing {len(parquet_files)} parquet files starting from {parquet_files[0]}"
        )  # noqa: T201

    if args.reverse:
        parquet_files = parquet_files[::-1]

    print(f"Found {len(parquet_files)} parquet files")  # noqa: T201

    output_dir = os.path.join(args.output_dir, "mistral_embeddings_bf16")
    os.makedirs(output_dir, exist_ok=True)

    for p_file in tqdm(parquet_files):
        filename = os.path.basename(p_file).split(".")[0]
        output_file = os.path.join(output_dir, f"{filename}.npy")

        if os.path.exists(output_file):
            print(f"Skipping {filename}")  # noqa: T201
            continue

        with pa.memory_map(p_file) as source:
            loaded_arrays = pa.RecordBatchStreamReader(source).read_all()

        df = loaded_arrays.to_pandas()

        file_embeddings = []

        captions = df["text"].values.tolist()

        for chunk_start in range(0, len(captions), args.batch_size):
            captions_chunk = captions[chunk_start : chunk_start + args.batch_size]  # noqa: E203

            batch_dict = tokenizer(
                captions_chunk,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                outputs = model(**batch_dict)
                embeddings = last_token_pool(
                    outputs.last_hidden_state, batch_dict["attention_mask"]
                )

            # normalize embeddings
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            embeddings = embeddings.float().cpu().detach().numpy()
            file_embeddings.extend(embeddings)

        file_embeddings = np.array(file_embeddings)
        np.save(output_file, file_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coyo_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_indices", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument(
        "--reverse", action="store_true", description="Reverse the order of the files"
    )

    args = parser.parse_args()
    main(args)
