import argparse
import os
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

__all__ = []


def main(args) -> None:
    """Main function."""

    model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")
    model = model.eval().cuda()

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

    output_dir = os.path.join(args.output_dir, "siglip_embeddings")
    os.makedirs(output_dir, exist_ok=True)

    for p_file in tqdm(parquet_files):
        filename = os.path.basename(p_file).split(".")[0]
        output_file = os.path.join(output_dir, f"{filename}.npy")

        if os.path.exists(output_file):
            print(f"Skipping {filename}")  # noqa: T201
            continue

        df = pd.read_parquet(p_file)

        file_embeddings = []

        captions = df["text"].values.tolist()

        for chunk_start in range(0, len(captions), args.batch_size):
            captions_chunk = captions[chunk_start : chunk_start + args.batch_size]  # noqa: E203

            tokens = tokenizer(captions_chunk).to("cuda")
            embeddings = model.encode_text(tokens)

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach().numpy()
            # print(f"Normalized embeddings shape: {embeddings.shape}")
            file_embeddings.extend(embeddings)

        file_embeddings = np.array(file_embeddings)
        np.save(output_file, file_embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_indices", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--cc12m_root", type=str, required=True)
    parser.add_argument(
        "--reverse", action="store_true", description="Reverse the order of the files"
    )

    args = parser.parse_args()
    main(args)
