import tarfile
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import requests
import torch
from tqdm import tqdm

__all__ = ["RetrievalDatabase", "download_retrieval_databases"]

RETRIEVAL_DATABASES_URLS = {
    "cc12m": {
        "url": "https://storage-cased.alessandroconti.me/cc12m.tar.gz",
        "cache_subdir": "./cc12m/vit-l-14/",
    },
    "cc12m_siglip": {
        "url": "",
        "cache_subdir": "./cc12m_siglip_384_urls/with_urls",
    },
    "cc12m_mistral": {
        "url": "",
        "cache_subdir": "./cc12m_mistral/with_urls",
    },
    "coyo_mistral": {
        "url": "",
        "cache_subdir": "./coyo_mistral/mistral",
    },
    "coyo_siglip": {
        "url": "",
        "cache_subdir": "./coyo_siglip/siglip",
    },
}


def download_retrieval_databases(cache_dir: str):
    """Download data if needed.

    Args:
        cache_dir (str): Path to cache directory. Defaults to "~/.cache/cased".
    """
    databases_path = Path(cache_dir, "databases")

    for name, items in RETRIEVAL_DATABASES_URLS.items():
        url = items["url"]
        database_path = Path(databases_path, name)
        if database_path.exists():
            print(f"Database: {name} OK")  # noqa: T201
            continue

        # download data
        target_path = Path(databases_path, name + ".tar.gz")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with requests.get(url, stream=True, timeout=10) as r:
            r.raise_for_status()
            total_bytes_size = int(r.headers.get("content-length", 0))
            chunk_size = 8192
            p_bar = tqdm(
                desc="Downloading cc12m index",
                total=total_bytes_size,
                unit="iB",
                unit_scale=True,
            )
            with open(target_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    p_bar.update(len(chunk))
            p_bar.close()

        # extract data
        tar = tarfile.open(target_path, "r:gz")
        tar.extractall(target_path.parent, filter="data")  # nosec
        tar.close()
        target_path.unlink()


class RetrievalDatabaseMetadataProvider:
    """Metadata provider for the retrieval database.

    Args:
        metadata_dir (str): Path to the metadata directory.
    """

    def __init__(self, metadata_dir: str):
        metadatas = [
            str(a) for a in sorted(Path(metadata_dir).glob("**/*")) if a.is_file()
        ]
        self.table = pa.concat_tables(
            [
                pa.ipc.RecordBatchFileReader(pa.memory_map(metadata, "r")).read_all()
                for metadata in metadatas
            ]
        )

    def get(self, ids: list) -> list[dict]:
        """Get the metadata for the given ids.

        Args:
            ids (list): List of ids.
        """
        columns = self.table.schema.names
        end_ids = [i + 1 for i in ids]
        t = pa.concat_tables(
            [self.table[start:end] for start, end in zip(ids, end_ids)]
        )
        return t.select(columns).to_pandas().to_dict("records")


class RetrievalDatabase:
    """Retrieval database.

    Args:
        database_name (str): Name of the database.
        cache_dir (str): Path to cache directory. Defaults to "~/.cache/cased".
    """

    def __init__(self, database_name: str, cache_dir: str):
        if database_name not in RETRIEVAL_DATABASES_URLS.keys():
            raise ValueError(
                f"Database name should be one of "
                f"{list(RETRIEVAL_DATABASES_URLS.keys())}, got {database_name}."
            )

        database_dir = Path(cache_dir) / "databases"
        database_dir = (
            database_dir / RETRIEVAL_DATABASES_URLS[database_name]["cache_subdir"]
        )
        self._database_dir = database_dir

        image_index_fp = Path(database_dir) / "image.index"
        text_index_fp = Path(database_dir) / "text.index"

        image_index = (
            faiss.read_index(
                str(image_index_fp), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            if image_index_fp.exists()
            else None
        )
        text_index = (
            faiss.read_index(
                str(text_index_fp), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
            )
            if text_index_fp.exists()
            else None
        )

        metadata_dir = str(Path(database_dir) / "metadata")
        metadata_provider = RetrievalDatabaseMetadataProvider(metadata_dir)

        self._image_index = image_index
        self._text_index = text_index
        self._metadata_provider = metadata_provider

    def _map_to_metadata(
        self, indices: list, distances: list, embs: list, num_images: int
    ):
        """Map the indices to metadata.

        Args:
            indices (list): List of indices.
            distances (list): List of distances.
            embs (list): List of results embeddings.
            num_images (int): Number of images.
        """
        results = []
        metas = self._metadata_provider.get(indices[:num_images])
        for key, (d, i, emb) in enumerate(zip(distances, indices, embs)):
            output = {}
            meta = None if key + 1 > len(metas) else metas[key]
            if meta is not None:
                output.update(self._meta_to_dict(meta))
            output["id"] = i.item()
            output["similarity"] = d.item()
            output["sample_z"] = emb.tolist()
            results.append(output)

        return results

    def _meta_to_dict(self, metadata: dict):
        """Convert metadata to dict.

        Args:
            metadata (dict): Metadata.
        """
        output = {}
        for k, v in metadata.items():
            if isinstance(v, bytes):
                v = v.decode()
            elif type(v).__module__ == np.__name__:
                v = v.item()
            output[k] = v
        return output

    def _get_connected_components(self, neighbors: dict):
        """Find connected components in a graph.

        Args:
            neighbors (dict): Dictionary of neighbors.
        """
        seen = set()

        def component(node):
            r = []
            nodes = {node}
            while nodes:
                node = nodes.pop()
                seen.add(node)
                nodes |= set(neighbors[node]) - seen
                r.append(node)
            return r

        u = []
        for node in neighbors:
            if node not in seen:
                u.append(component(node))
        return u

    def _deduplicate_embeddings(
        self, embeddings: np.ndarray, threshold: float = 0.94
    ) -> set:  # noqa: CCE001
        """Deduplicate embeddings.

        Args:
            embeddings (np.matrix): Embeddings to deduplicate.
            threshold (float): Threshold to use for deduplication. Default is 0.94.
        """
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        l, _, indices = index.range_search(embeddings, threshold)  # noqa: E741

        same_mapping = defaultdict(list)

        for i in range(embeddings.shape[0]):
            start = l[i]
            end = l[i + 1]
            for j in indices[start:end]:
                same_mapping[int(i)].append(int(j))

        groups = self._get_connected_components(same_mapping)
        non_uniques = set()
        for g in groups:
            for e in g[1:]:
                non_uniques.add(e)

        return set(list(non_uniques))

    def query(
        self,
        images_z: torch.tensor,
        modality: str = "text",
        num_samples: int = 10,
        normalize=True,
    ) -> tuple:
        """Query the database.

        Args:
            query (np.matrix): Query to search.
            modality (str): Modality to search. One of `image` or `text`. Default to `text`.
            num_samples (int): Number of samples to return. Default is 40.
        """
        if normalize:
            images_z = images_z / images_z.norm(dim=-1, keepdim=True)
        images_z = images_z.cpu().detach().numpy().tolist()

        if isinstance(images_z[0], float):
            images_z = [images_z]

        query = np.matrix(images_z).astype("float32")

        index = self._image_index if modality == "image" else self._text_index

        distances, indices, embeddings = index.search_and_reconstruct(
            query, num_samples
        )
        results = [indices[i] for i in range(len(indices))]

        nb_results = [np.where(r == -1)[0] for r in results]
        total_distances = []
        total_indices = []
        total_embeddings = []
        for i in range(len(results)):
            num_res = nb_results[i][0] if len(nb_results[i]) > 0 else len(results[i])

            result_indices = results[i][:num_res]
            result_distances = distances[i][:num_res]
            result_embeddings = embeddings[i][:num_res]

            # normalise embeddings
            l2 = np.atleast_1d(np.linalg.norm(result_embeddings, 2, -1))
            l2[l2 == 0] = 1
            result_embeddings = result_embeddings / np.expand_dims(l2, -1)

            # deduplicate embeddings
            local_indices_to_remove = self._deduplicate_embeddings(result_embeddings)
            indices_to_remove = set()
            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])

            curr_indices = []
            curr_distances = []
            curr_embeddings = []
            for ind, dis, emb in zip(
                result_indices, result_distances, result_embeddings
            ):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    curr_indices.append(ind)
                    curr_distances.append(dis)
                    curr_embeddings.append(emb)

            total_indices.append(curr_indices)
            total_distances.append(curr_distances)
            total_embeddings.append(curr_embeddings)

        if len(total_distances) == 0:
            return []

        total_results = []
        for i in range(len(total_distances)):
            results = self._map_to_metadata(
                total_indices[i], total_distances[i], total_embeddings[i], num_samples
            )
            total_results.append(results)

        captions = [[r["caption"] for r in result] for result in total_results]
        similarities = [[r["similarity"] for r in result] for result in total_results]

        # return captions, image_urls, image_keys, similarities
        return captions, similarities

    def query_by_embedding(
        self, query, modality: str = "text", num_samples: int = 10
    ) -> tuple:
        """
        Query the database by embedding.
        Args:
            query (np.matrix): Query to search.
            modality (str): Modality to search. One of `image` or `text`. Default to `text`.
            num_samples (int): Number of samples to return. Default is 40.
        """
        index = self._image_index if modality == "image" else self._text_index

        if isinstance(query, torch.Tensor):
            query = query.detach().cpu().numpy()

        query = np.matrix(query).astype("float32")

        distances, indices, embeddings = index.search_and_reconstruct(
            query, num_samples
        )
        results = [indices[i] for i in range(len(indices))]

        nb_results = [np.where(r == -1)[0] for r in results]
        total_distances = []
        total_indices = []
        total_embeddings = []
        for i in range(len(results)):
            num_res = nb_results[i][0] if len(nb_results[i]) > 0 else len(results[i])

            result_indices = results[i][:num_res]
            result_distances = distances[i][:num_res]
            result_embeddings = embeddings[i][:num_res]

            # normalise embeddings
            l2 = np.atleast_1d(np.linalg.norm(result_embeddings, 2, -1))
            l2[l2 == 0] = 1
            result_embeddings = result_embeddings / np.expand_dims(l2, -1)

            # deduplicate embeddings
            local_indices_to_remove = self._deduplicate_embeddings(result_embeddings)
            indices_to_remove = set()
            for local_index in local_indices_to_remove:
                indices_to_remove.add(result_indices[local_index])

            curr_indices = []
            curr_distances = []
            curr_embeddings = []
            for ind, dis, emb in zip(
                result_indices, result_distances, result_embeddings
            ):
                if ind not in indices_to_remove:
                    indices_to_remove.add(ind)
                    curr_indices.append(ind)
                    curr_distances.append(dis)
                    curr_embeddings.append(emb)

            total_indices.append(curr_indices)
            total_distances.append(curr_distances)
            total_embeddings.append(curr_embeddings)

        if len(total_distances) == 0:
            return []

        total_results = []
        for i in range(len(total_distances)):
            results = self._map_to_metadata(
                total_indices[i], total_distances[i], total_embeddings[i], num_samples
            )
            total_results.append(results)

        captions = [[r["caption"] for r in result] for result in total_results]
        similarities = [[r["similarity"] for r in result] for result in total_results]

        # return captions, image_urls, image_keys, similarities
        return captions, similarities
