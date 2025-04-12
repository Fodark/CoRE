import argparse
import os

import open_clip
import pandas as pd
import rootutils
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.retrieval_dataset import DatasetRetrieval  # noqa: E402
from src.retrieval_database import RetrievalDatabase  # noqa: E402

__all__ = []


def load_retrieval_databases(db: str):
    """Load the retrieval databases."""
    print("Loading retrieval databases...")  # noqa: T201
    img2txt_database = RetrievalDatabase(f"{db}_siglip", ".cache/")

    return img2txt_database


def embed_image(image: Image.Image, transform, model, normalize=False, device="cuda"):
    """Embed an image using a model."""
    with torch.no_grad():
        dino_preprocessed = transform(image).unsqueeze(0).to(device)
        dino_embedding = model(dino_preprocessed)
        if normalize:
            dino_embedding = dino_embedding / dino_embedding.norm(dim=-1, keepdim=True)

    return dino_embedding.cpu()


def embed_image_siglip(image: Image.Image, preprocess_val, model, device):
    """Embed an image using the SigLIP model."""
    with torch.no_grad():
        image = preprocess_val(image).unsqueeze(0).to(device)
        image_embedding = model.encode_image(image).squeeze(0)
        image_embedding = F.normalize(image_embedding, p=2, dim=-1)

    return image_embedding.cpu()


def load_siglip(device):
    """Load the SigLIP model."""
    print("Loading SigLIP...")  # noqa: T201
    model, _, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
    )
    model.eval()
    model.to(device)
    tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")

    return model, preprocess_val, tokenizer


def get_retrieval_siglip(
    query, img2txt_database, siglip_model, siglip_preprocess_val, tokenizer
):
    """Get retrieval results from the SigLIP model."""
    with torch.no_grad():
        siglip_preprocessed = siglip_preprocess_val(query).unsqueeze(0)
        siglip_embedding = siglip_model.encode_image(siglip_preprocessed).squeeze(0)
        siglip_embedding = siglip_embedding / siglip_embedding.norm(
            dim=-1, keepdim=True
        )

        siglip_retrieval = img2txt_database.query(siglip_embedding, modality="text")
        captions, similarities = siglip_retrieval
        text_similarities = similarities

    retrieved_text_embeddings = []

    for _, caption in enumerate(captions):
        caption_tokens = tokenizer(caption, context_length=siglip_model.context_length)
        with torch.no_grad():
            caption_embedding = siglip_model.encode_text(caption_tokens).squeeze(0)
        caption_embedding = caption_embedding / caption_embedding.norm(
            dim=-1, keepdim=True
        )
        retrieved_text_embeddings.append(caption_embedding)

    retrieved_text_embeddings = torch.stack(retrieved_text_embeddings)

    return captions, text_similarities, retrieved_text_embeddings


def get_retrieval_siglip_by_embedding(embedding, database):
    """Get retrieval results from the SigLIP model using an embedding."""
    siglip_retrieval = database.query_by_embedding(embedding, modality="text")
    captions, similarities = siglip_retrieval

    return captions, similarities


def load_datasets(
    dataset_name, preprocess_fn_siglip=None, siglip_model=None, split="train"
):
    """Load the datasets."""
    print("Loading datasets...")  # noqa: T201
    dataset = DatasetRetrieval(dataset_name, split, preprocess_fn_siglip, siglip_model)
    class_mapping_file = f"artifacts/data/{dataset_name}/class_mapping.csv"
    class_mapping = pd.read_csv(class_mapping_file)
    class_mapping_dict = class_mapping.set_index("id")["name"].to_dict()

    return dataset, class_mapping_dict, dataset_name


def main(dataset_name, split, db):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = f"retrieved_embeddings/{db}/{dataset_name}/{split}"
    os.makedirs(out_dir, exist_ok=True)

    siglip_model, siglip_preprocess_val, tokenizer = load_siglip(device)
    img2txt_database = load_retrieval_databases(db)

    dataset, class_mapping, dataset_name = load_datasets(
        dataset_name, siglip_preprocess_val, siglip_model, split
    )

    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    sample_idx = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        image_paths = batch["path"]
        label_idxs = batch["label"]
        class_names = [class_mapping[label_idx.item()] for label_idx in label_idxs]
        original_embeddings_siglip = batch["siglip_emb"]

        query_captions, original_similarities = get_retrieval_siglip_by_embedding(
            original_embeddings_siglip, img2txt_database
        )

        for idx in range(len(image_paths)):
            filename = image_paths[idx]
            query_results = {
                "filename": filename,
                "classname": class_names[idx],
                "query_embedding": original_embeddings_siglip[idx],
                "query_retrieved_captions": query_captions[idx],
                "query_retrieved_captions_similarities": original_similarities[idx],
            }
            torch.save(query_results, f"{out_dir}/{sample_idx:05d}.pt")
            sample_idx += 1


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_name", type=str, required=True)
    argparser.add_argument("--split", type=str, required=True)
    argparser.add_argument("--db", type=str, required=True)
    args = argparser.parse_args()

    torch.multiprocessing.set_start_method("spawn")
    main(args.dataset_name, args.split, args.db)
