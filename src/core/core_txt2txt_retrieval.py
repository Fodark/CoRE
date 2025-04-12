import argparse
import json
from functools import partial

import open_clip
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.retrieval_database import RetrievalDatabase  # noqa

__all__ = []


def load_retrieval_databases(database_name: str = "coyo_mistral"):
    """Load the retrieval databases."""
    print("Loading retrieval databases...")  # noqa: T201
    database = RetrievalDatabase(database_name, ".cache/")

    return database


def last_token_pool(
    last_hidden_states: torch.tensor, attention_mask: torch.tensor
) -> torch.Tensor:
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


def load_siglip():
    """Load the SigLIP model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading SigLIP...")  # noqa: T201
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")

    embed_fn = partial(get_captions, model=model, tokenizer=tokenizer)

    return embed_fn


def load_mistral():
    """Load the Mistral model."""
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")
    model = model.eval()
    max_length = 4096

    embed_fn = partial(
        get_captions_mistral, model=model, tokenizer=tokenizer, max_length=max_length
    )

    return embed_fn
    # return model, tokenizer


def get_captions_mistral(
    prompt, model, tokenizer, database, max_length
) -> tuple[list[str], list[str]]:
    """Get captions and urls from the database."""
    batch_dict = tokenizer(
        [prompt],
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    outputs = model(**batch_dict)
    embeddings = last_token_pool(
        outputs.last_hidden_state, batch_dict["attention_mask"]
    )
    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach()
    captions, similarities = database.query(embeddings)

    return captions[0], similarities[0]


def get_captions(
    prompt: str, model, tokenizer, database
) -> tuple[list[str], list[str]]:
    """Get captions and urls from the database."""

    tokens = tokenizer([prompt])
    with torch.no_grad():
        embeddings = model.encode_text(tokens)

    embeddings = F.normalize(embeddings, p=2, dim=1).cpu().detach()
    captions, similarities = database.query(embeddings)

    return captions[0], similarities[0]


dataset_data = {
    "circuits": {
        "classname_file": "artifacts/data/circuits-complete/class_mapping.csv",
        "classname_file_common": None,
        "suffix": "common",
        "templates": ["A photo of a {}.", "A circuit diagram of a {}."],
    },
    "inat": {
        "classname_file": "artifacts/data/inaturalist/class_mapping.csv",
        "classname_file_common": "artifacts/data/inaturalist/class_mapping_common.csv",
        "suffix": "premerged",
        "templates": ["A photo of a {}."],
    },
    "ham": {
        "classname_file": "artifacts/data/ham10000/class_mapping.csv",
        "classname_file_common": None,
        "suffix": "common",
        "templates": ["A photo of a {}.", "A skin lesion of a {}."],
    },
}

models = {"siglip": load_siglip(), "mistral": load_mistral()}


def get_dataset_files(dataset_name):
    """Get the dataset files."""
    return dataset_data[dataset_name]


def main(dataset_name, model, database_name):
    """Main function."""
    dataset_files = get_dataset_files(dataset_name)

    classnames_file: str = dataset_files["classname_file"]
    classnames: list[str] = pd.read_csv(classnames_file)["name"].tolist()

    classname_file_common: str | None = dataset_files["classname_file_common"]
    classnames_common: list[str] | None = None
    if classname_file_common is not None:
        classnames_common: list = pd.read_csv(classname_file_common)["name"].tolist()

    suffix: str = dataset_files["suffix"]
    templates: list[str] = dataset_files["templates"]

    print(f"#classnames: {len(classnames)}")  # noqa: T201
    log_file = f"results/{model}_index_{dataset_name}_{suffix}.json"

    results = {}

    embed_fn = models[model]

    database = load_retrieval_databases(database_name)

    for template in templates:
        results[template] = {}

        for idx, classname in enumerate(classnames):
            if classnames_common is not None:
                prompt = template.format(f"{classname} ({classnames_common[idx]})")
            else:
                prompt = template.format(classname)
            captions, similarities = embed_fn(database=database, prompt=prompt)

            if results[template].get(classname) is None:
                results[template][classname] = {
                    "captions": captions,
                    "similarities": similarities,
                }
            else:
                results[template][f"{classname} ({classnames_common[idx]})"] = {
                    "captions": captions,
                    "similarities": similarities,
                }

    tqdm.write(f"Saving results for {len(results[template])} classes to {log_file}.")
    with open(log_file, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--database_name", type=str, default="coyo_mistral")
    parser.add_argument("--model", type=str, required=True)

    args = parser.parse_args()

    main(args.dataset, args.model, args.database_name)
