import enum
import glob
import os
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def pad_image(image: Image.Image) -> Image.Image:
    """A function to pad an image to square."""
    # pad to square using PIL
    width, height = image.size
    if width > height:
        new_width = width
        new_height = width
    else:
        new_width = height
        new_height = height
    new_im = Image.new("RGB", (new_width, new_height))
    new_im.paste(image, ((new_width - width) // 2, (new_height - height) // 2))

    return new_im


class DatasetPIL(Dataset):
    """Generic dataset for any type of folder-based class dataset.

    Returns PIL images instead of `torch.Tensor`
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        reduced: bool = False,
        n_classes: int | None = None,
        shots_per_class: int | None = None,
    ):
        data_root: Path = Path("data")
        data_root = data_root / dataset / split
        self.data_root = data_root

        self.dataset_name = dataset
        self.split = split
        self.data: list[tuple[Path, int]] = []
        self.class_mapping: dict[int, str] = {}
        self.class_mapping_common: dict[int, str] = {}
        self.class_names: list[str] = []
        self.class_names_common: list[str] = []

        self.reduced = reduced
        self.n_classes = n_classes
        self.shots_per_class = shots_per_class

        self._setup()

    def _setup(self) -> None:  # noqa: CCE001
        """Load the data stored in `self.data_root`"""

        if not self.reduced:
            class_mapping_file = f"artifacts/data/{self.dataset_name}/class_mapping.csv"
            class_mapping_common = (
                f"artifacts/data/{self.dataset_name}/class_mapping_common.csv"
            )
        else:
            class_mapping_file = (
                f"artifacts/data/{self.dataset_name}/class_mapping_reduced.csv"
            )
            class_mapping_common = (
                f"artifacts/data/{self.dataset_name}/class_mapping_common_reduced.csv"
            )

        samples_file = Path(f"artifacts/data/{self.dataset_name}")

        if self.split == "train":
            if self.n_classes is not None and self.shots_per_class is not None:
                samples_file = (
                    samples_file
                    / f"{self.split}_{self.n_classes}way_{self.shots_per_class}shot.csv"
                )
            else:
                samples_file = samples_file / f"{self.split}.csv"
        else:
            samples_file = samples_file / f"{self.split}.csv"

        class_mapping_df = pd.read_csv(class_mapping_file)
        self.class_mapping = dict(zip(class_mapping_df["id"], class_mapping_df["name"]))

        if os.path.exists(class_mapping_common):
            class_mapping_common_df = pd.read_csv(class_mapping_common)
            self.class_mapping_common = dict(
                zip(class_mapping_common_df["id"], class_mapping_common_df["name"])
            )
        else:
            self.class_mapping_common = self.class_mapping

        class_names = list(self.class_mapping.values())
        self.class_names = class_names

        class_names_common = list(self.class_mapping_common.values())
        self.class_names_common = class_names_common

        samples_df = pd.read_csv(samples_file)

        for _, row in samples_df.iterrows():
            img_path = self.data_root / row["filename"]
            label = row["class_id"]
            self.data.append((img_path, int(label)))

    def __str__(self) -> str:
        return f"{self.dataset_name}-{self.split}"

    @property
    def prompt(self) -> str:
        """Returns the correct CLIP prompt based on the dataset loaded."""
        if "circuits" in str(self.data_root):
            return "A circuit diagram of a {}."
        elif "egg" in str(self.data_root):
            return "A parasite egg of {}."
        elif "ham" in str(self.data_root):
            return "A skin lesion of {}."
        elif "inaturalist" in str(self.data_root):
            return "A photo of a {}."
        else:
            raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, Path]:
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        return img, label, img_path


class DatasetRetrieval(Dataset):
    """Generic dataset for any type of folder-based class dataset.

    Returns PIL images instead of `torch.Tensor`
    """

    def __init__(
        self,
        dataset: str,
        split: str,
        preprocess_fn_siglip,
        siglip_model,
        data_root: str = "data",
    ):
        data_root: Path = Path(data_root)
        data_root = data_root / dataset / split
        self.data_root = data_root

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.preprocess_fn_siglip = preprocess_fn_siglip
        self.siglip_model = siglip_model

        self.dataset_name = dataset
        self.split = split
        self.data: list[tuple[Path, int]] = []
        self.class_mapping: dict[int, str] = {}
        self.class_mapping_common: dict[int, str] = {}
        self.class_names: list[str] = []
        self.class_names_common: list[str] = []

        self._setup()

    def _setup(self) -> None:  # noqa: CCE001
        """Load the data stored in `self.data_root`"""

        class_mapping_file = f"artifacts/data/{self.dataset_name}/class_mapping.csv"
        class_mapping_common = (
            f"artifacts/data/{self.dataset_name}/class_mapping_common.csv"
        )

        samples_file = Path(f"artifacts/data/{self.dataset_name}")
        samples_file = samples_file / f"{self.split}.csv"
        samples_file = str(samples_file)

        class_mapping_df = pd.read_csv(class_mapping_file)
        self.class_mapping = dict(zip(class_mapping_df["id"], class_mapping_df["name"]))

        if os.path.exists(class_mapping_common):
            class_mapping_common_df = pd.read_csv(class_mapping_common)
            self.class_mapping_common = dict(
                zip(class_mapping_common_df["id"], class_mapping_common_df["name"])
            )
        else:
            self.class_mapping_common = self.class_mapping

        class_names = list(self.class_mapping.values())
        self.class_names = class_names

        class_names_common = list(self.class_mapping_common.values())
        self.class_names_common = class_names_common

        samples_df = pd.read_csv(samples_file)

        for _, row in samples_df.iterrows():
            img_path = self.data_root / row["filename"]
            label = row["class_id"]
            self.data.append((img_path, int(label)))

    def __str__(self) -> str:
        return f"{self.dataset_name}-{self.split}"

    @property
    def prompt(self) -> str:
        """Returns the correct CLIP prompt based on the dataset loaded."""
        if "circuits" in str(self.data_root):
            return "A circuit diagram of a {}."
        elif "egg" in str(self.data_root):
            return "A parasite egg of {}."
        elif "ham" in str(self.data_root):
            return "A skin lesion of {}."
        elif "inaturalist" in str(self.data_root):
            return "A photo of a {}."
        else:
            raise NotImplementedError()

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, Path]:
        img_path, label = self.data[idx]

        # check if img_path misses the extension
        if not img_path.suffix:
            img_path = img_path.with_suffix(".jpg")

        img = Image.open(img_path).convert("RGB")

        with torch.no_grad():
            siglip_pre = (
                self.preprocess_fn_siglip(img).detach().unsqueeze(0).to(self.device)
            )
            siglip_emb = (
                self.siglip_model.encode_image(siglip_pre).detach().cpu().squeeze()
            )
            siglip_emb /= siglip_emb.norm(dim=-1, keepdim=True)

        sample_dict = {"path": str(img_path), "label": label, "siglip_emb": siglip_emb}

        return sample_dict


# define an enum of 3 types of names to use, between scientific, common and merged
class ZeroshotType(str, enum.Enum):
    """Enum for the zeroshot type."""

    SCIENTIFIC = "scientific"
    COMMON = "common"
    MERGED = "merged"


class DatasetEmbedding(Dataset):
    """
    Loads retrieved embeddings from image-to-text retrieval
    """

    # data = []
    variants: list[str] = ["query_embedding", "query_retrieved_captions", "beta_merged"]

    def __init__(
        self,
        dataset_name: str,
        retrieval_database: str,
        tokenizer,
        clip_model,
        split: str = "test",
        temperature: float = 1.0,
        beta: float = 0.0,
    ):
        self.dataset_name = dataset_name  # if not self.reduced else "inat_reduced"
        self.retrieval_database = retrieval_database
        self.split = split
        self.temperature = temperature
        self.beta = beta
        self.tokenizer = tokenizer
        self.clip_model = clip_model

        self.features_folder = Path(
            "data", "feats", retrieval_database, dataset_name, split
        )
        os.makedirs(self.features_folder, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.retrieved_root = os.path.join(
            "retrieved_embeddings", retrieval_database, dataset_name, split
        )

        self._setup()

    def _setup(self) -> None:  # noqa: CCE001
        """Load the data stored in `self.data_root`"""
        self.data = []

        class_mapping_file = f"artifacts/data/{self.dataset_name}/class_mapping.csv"
        class_mapping_common = (
            f"artifacts/data/{self.dataset_name}/class_mapping_common.csv"
        )

        class_mapping_df = pd.read_csv(class_mapping_file)
        self.class_mapping = dict(zip(class_mapping_df["id"], class_mapping_df["name"]))

        if os.path.exists(class_mapping_common):
            class_mapping_common_df = pd.read_csv(class_mapping_common)
            self.class_mapping_common = dict(
                zip(class_mapping_common_df["id"], class_mapping_common_df["name"])
            )
        else:
            self.class_mapping_common = self.class_mapping

        class_names = list(self.class_mapping.values())
        self.class_names = class_names

        class_names_common = list(self.class_mapping_common.values())
        self.class_names_common = class_names_common

        caption_files = glob.glob(f"{self.retrieved_root}/*.pt")

        for caption_file in caption_files:
            captions_dict: dict = torch.load(caption_file)
            self.data.append(captions_dict)

    def build_zeroshot_weights(  # noqa: CCE001
        self,
        template: str,
        zeroshot_type: ZeroshotType,
        retrieved_zeroshot_file: str | None = None,
        merge: bool = False,
        alpha: float = 0.5,
    ) -> torch.FloatTensor:
        """Build the zero-shot weights for the dataset."""
        if retrieved_zeroshot_file is not None:
            retrieved_embeddings = torch.load(
                retrieved_zeroshot_file, map_location="cpu"
            )
            retrieved_embeddings = retrieved_embeddings[-self.num_classes :, :]  # noqa: E203

        template_formatted = (
            template.format("")
            .strip()
            .replace(" ", "_")
            .replace("/", "_")
            .replace(".", "")
            .lower()
        )

        zeroshot_path = os.path.join(
            self.features_folder,
            f"{template_formatted}_{zeroshot_type.value}_zeroshot_weights.pt",
        )

        if os.path.exists(zeroshot_path):
            zeroshot_embeddings = torch.load(zeroshot_path)

            if merge:
                zeroshot_embeddings = (
                    alpha * retrieved_embeddings + (1 - alpha) * zeroshot_embeddings
                )
                zeroshot_embeddings = zeroshot_embeddings / zeroshot_embeddings.norm(
                    dim=-1, keepdim=True
                )

            return zeroshot_embeddings

        if (
            zeroshot_type == ZeroshotType.SCIENTIFIC
            or zeroshot_type == ZeroshotType.MERGED
        ):
            class_names = self.class_names
        elif zeroshot_type == ZeroshotType.COMMON:
            class_names = self.class_names_common

        if zeroshot_type != ZeroshotType.MERGED:
            formatted_class_names: list[str] = [
                template.format(class_name) for class_name in class_names
            ]
            tokens = self.tokenizer(formatted_class_names).to(self.device)
            with torch.no_grad():
                zeroshot_embeddings: torch.FloatTensor = (
                    self.clip_model.encode_text(tokens).detach().cpu()
                )
                zeroshot_embeddings = zeroshot_embeddings / zeroshot_embeddings.norm(
                    dim=-1, keepdim=True
                )
        else:
            formatted_class_names_scientific: list[str] = [
                template.format(class_name) for class_name in self.class_names
            ]
            formatted_class_names_common: list[str] = [
                template.format(class_name) for class_name in self.class_names_common
            ]
            tokens_scientific = self.tokenizer(formatted_class_names_scientific).to(
                self.device
            )
            tokens_common = self.tokenizer(formatted_class_names_common).to(self.device)
            with torch.no_grad():
                embeddings_scientific: torch.FloatTensor = (
                    self.clip_model.encode_text(tokens_scientific).detach().cpu()
                )
                embeddings_scientific = (
                    embeddings_scientific
                    / embeddings_scientific.norm(dim=-1, keepdim=True)
                )
                embeddings_common: torch.FloatTensor = (
                    self.clip_model.encode_text(tokens_common).detach().cpu()
                )
                embeddings_common = embeddings_common / embeddings_common.norm(
                    dim=-1, keepdim=True
                )
                zeroshot_embeddings = (embeddings_scientific + embeddings_common) / 2
                zeroshot_embeddings = zeroshot_embeddings / zeroshot_embeddings.norm(
                    dim=-1, keepdim=True
                )

        if not os.path.exists(zeroshot_path):
            torch.save(zeroshot_embeddings, zeroshot_path)
            print(f"Zero-shot weights saved to {zeroshot_path}")  # noqa: T201

        if zeroshot_embeddings.shape[0] != len(class_names):
            raise ValueError("Embeddings and class names must have the same length")

        if merge:
            # print("Merging with retrieved embeddings...")  # noqa: T201
            zeroshot_embeddings = (
                alpha * retrieved_embeddings + (1 - alpha) * zeroshot_embeddings
            )
            zeroshot_embeddings = zeroshot_embeddings / zeroshot_embeddings.norm(
                dim=-1, keepdim=True
            )

        return zeroshot_embeddings

    def __str__(self) -> str:
        return f"{self.dataset_name}-captions"

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        # print("Getting item", idx, len(self.data))
        # img_path, label = self.data[idx]
        sample_dict = self.data[idx]
        img_path = sample_dict["filename"]
        classname = sample_dict["classname"]
        try:
            label = self.class_names.index(classname)
        except ValueError:
            label = self.class_names_common.index(classname)

        label = torch.tensor(label, dtype=torch.long)
        basename_without_ext = os.path.splitext(os.path.basename(img_path))[0]

        sample = {
            "path": img_path,
            "label": label,
            "query_embedding": sample_dict["query_embedding"],
        }

        for variant in self.variants[1:]:
            if variant == "beta_merged":
                # beta_merged = beta * original + (1 - beta) * image
                beta = self.beta
                original_embedding = sample["query_embedding"]
                image_embedding = sample["query_retrieved_captions"]
                beta_merged_embedding = (1 - beta) * original_embedding + (
                    beta
                ) * image_embedding
                beta_merged_embedding = (
                    beta_merged_embedding
                    / beta_merged_embedding.norm(dim=-1, keepdim=True)
                )
                sample[variant] = beta_merged_embedding
                continue

            text_variant_embedding_path = os.path.join(
                self.features_folder,
                f"{variant}_features_t_{self.temperature}",
                f"{basename_without_ext}.pt",
            )
            # create intermediate folder if it does not exist
            os.makedirs(os.path.dirname(text_variant_embedding_path), exist_ok=True)
            if os.path.exists(text_variant_embedding_path):
                caption_embedding = torch.load(text_variant_embedding_path)
            else:
                captions = sample_dict[variant]
                similarities = sample_dict[f"{variant}_similarities"]

                caption_tokens = self.tokenizer(captions).to(self.device)
                caption_embedding = (
                    self.clip_model.encode_text(caption_tokens).detach().cpu()
                )
                caption_embedding = caption_embedding / caption_embedding.norm(
                    dim=-1, keepdim=True
                )

                caption_embedding *= (
                    torch.tensor(similarities) * self.temperature
                ).unsqueeze(-1)
                caption_embedding = caption_embedding.sum(dim=0)
                caption_embedding = caption_embedding / caption_embedding.norm(
                    dim=-1, keepdim=True
                )
                caption_embedding = caption_embedding.squeeze()
                torch.save(caption_embedding, text_variant_embedding_path)

            sample[variant] = caption_embedding

        return sample


__all__ = [DatasetPIL, DatasetRetrieval, DatasetEmbedding, ZeroshotType]
