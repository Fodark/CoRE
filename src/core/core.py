import argparse

import open_clip
import rootutils
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from tqdm import tqdm

rootutils.setup_root(".", indicator=".project-root", pythonpath=True)

from src.data.retrieval_dataset import DatasetEmbedding, ZeroshotType  # noqa

load_dotenv()

retrieved_weights_data_circuits: list[tuple[str, str, str]] = [
    (
        "coyo_circuits_weighted_001_common_zeroshot_weights.pt",
        "A photo of a",
        "0.01",
    ),
]

retrieved_weights_data_ham: list[tuple[str, str, str]] = [
    (
        "coyo_ham_weighted_001_common_zeroshot_weights.pt",
        "A photo of a",
        "0.01",
    ),
]

retrieved_weights_data_inat_reduced: list[tuple[str, str, str]] = [
    (
        "coyo_inat_weighted_001_premerged_zeroshot_weights.pt",
        "A photo of a",
        "0.01",
    ),
]


def load_dataset(
    dataset_name: str,
    db: str,
    split: str,
    temperature: float,
    beta: float,
    model,
    preprocess,
    tokenizer,
    generic: bool = False,
):
    """
    Load the dataset and the corresponding template and weights.
    Args:
        dataset_name (str): The name of the dataset.
        db (str): The database to use.
        split (str): The split to use.
        temperature (float): The temperature to use.
        beta (float): The beta to use.
        model: The model to use.
        preprocess: The preprocess to use.
        tokenizer: The tokenizer to use.
        generic (bool): Whether to use the generic template or not.
    """
    templates = {
        "circuits": "A circuit diagram of a {}.",
        "ham10000": "A skin lesion of a {}.",
        "inaturalist": "A {}.",
    }
    template_generic = "A photo of a {}."
    weights = {
        "circuits": retrieved_weights_data_circuits,
        "ham10000": retrieved_weights_data_ham,
        "inaturalist": retrieved_weights_data_inat_reduced,
    }

    dataset = DatasetEmbedding(
        dataset_name=dataset_name,
        retrieval_database=db,
        split=split,
        tokenizer=tokenizer,
        clip_model=model,
        temperature=temperature,
        beta=beta,
    )

    if generic:
        return dataset, template_generic, weights[dataset_name]

    return dataset, templates[dataset_name], weights[dataset_name]


def load_siglip(model_name: str):
    """
    Load the SigLIP model and tokenizer.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess_val, tokenizer


def objective(dataset_name, db, split, **kwargs):
    """
    Objective function to evaluate the model on the dataset.
    Args:
        dataset_name (str): The name of the dataset.
        db (str): The database to use.
        split (str): The split to use.
        kwargs: Additional arguments for the function.
    """
    global model, preprocess, tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variant = "beta_merged"

    dataset_temperature = kwargs.get("dataset_t", 0.01)
    weight_indices = kwargs.get("weight_idx", None)
    if weight_indices is None:
        weight_indices = [0, 1, 2, 3, 4]
    alphas = kwargs.get("alpha", None)
    if alphas is None:
        alphas = [x / 10.0 for x in list(range(1, 11))]
    betas = kwargs.get("beta", None)
    if betas is None:
        betas = [x / 10.0 for x in list(range(1, 11))]

    # if weight_index is a int, then make it a list
    if isinstance(weight_indices, int):
        weight_indices = [weight_indices]

    # same for alpha and beta
    if isinstance(alphas, float):
        alphas = [alphas]

    if isinstance(betas, float):
        betas = [betas]

    best_acc = 0.0

    for weight_index in tqdm(weight_indices, leave=False):
        for alpha in tqdm(alphas, leave=False):
            for beta in tqdm(betas, leave=False):
                dataset, template, retrieved_weights = load_dataset(
                    dataset_name=dataset_name,
                    db=db,
                    split=split,
                    temperature=dataset_temperature,
                    beta=beta,
                    model=model,
                    preprocess=preprocess,
                    tokenizer=tokenizer,
                    generic=kwargs.get("generic", False),
                )
                weight_file, _, _ = retrieved_weights[weight_index]

                accuracy = Accuracy(task="multiclass", num_classes=dataset.num_classes)
                accuracy_at_5 = Accuracy(
                    task="multiclass", num_classes=dataset.num_classes, top_k=5
                )

                zeroshot_weights = dataset.build_zeroshot_weights(
                    template,
                    ZeroshotType.MERGED,
                    retrieved_zeroshot_file=f"retrieved_zeroshot_files/{weight_file}",
                    merge=True,
                    alpha=alpha,
                )
                zeroshot_weights = zeroshot_weights.to(device)

                dataloader = DataLoader(dataset, batch_size=128, num_workers=4)

                for batch in tqdm(dataloader, disable=True):
                    groundtruth = batch["label"]

                    embedding = batch[variant]
                    embedding = embedding.to(device)
                    logits = embedding @ zeroshot_weights.T
                    logits = logits.cpu()

                    accuracy.update(logits, groundtruth)
                    accuracy_at_5.update(logits, groundtruth)

                accuracy_val = accuracy.compute().item()
                accuracy_at_5_val = accuracy_at_5.compute().item()

                if accuracy_val > best_acc:
                    tqdm.write(
                        f"{dataset_name=} | {db=} | {alpha=} | {beta=} | {weight_index=} | {dataset_temperature=} || {accuracy_val=:.4f} {accuracy_at_5_val=:.4f}"
                    )
                    best_acc = accuracy_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--db", type=str, required=True)
    parser.add_argument("--split", type=str, required=True)

    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--dataset_t", type=float, default=None)
    parser.add_argument("--weight_idx", type=int, default=None)
    parser.add_argument("--generic", action="store_true")

    args = parser.parse_args()

    torch.multiprocessing.set_start_method("spawn")
    global model, preprocess, tokenizer
    model, preprocess, tokenizer = load_siglip("hf-hub:timm/ViT-SO400M-14-SigLIP-384")

    objective(
        args.dataset,
        args.db,
        args.split,
        alpha=args.alpha,
        beta=args.beta,
        dataset_t=args.dataset_t,
        weight_idx=args.weight_idx,
        generic=args.generic,
    )
