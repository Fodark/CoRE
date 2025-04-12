import argparse
import json

import open_clip
import rootutils
import torch

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)


def main(model: str, dataset: str, suffix: str):
    """Main function to load the model and process the results."""
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "hf-hub:timm/ViT-SO400M-14-SigLIP-384"
    )
    tokenizer = open_clip.get_tokenizer("hf-hub:timm/ViT-SO400M-14-SigLIP-384")

    results_file = f"results/{model}_index_{dataset}_{suffix}.json"

    results = json.load(open(results_file))

    prefixes = list(results.keys())

    for key in prefixes:
        classnames = list(results[key].keys())

        zeroshot_weights_naive = []
        zeroshot_weights_weighted_1 = []
        zeroshot_weights_weighted_01 = []
        zeroshot_weights_weighted_001 = []
        zeroshot_weights_weighted_10 = []
        zeroshot_weights_weighted_100 = []

        for classname in classnames:
            captions = results[key][classname]["captions"]
            similarities = torch.tensor(results[key][classname]["similarities"])
            print(f"Classname: {classname}, Captions: {captions}")  # noqa: T201

            captions_tokens = tokenizer(captions, context_length=model.context_length)
            with torch.no_grad(), torch.cuda.amp.autocast():
                captions_embeddings = model.encode_text(captions_tokens)
                captions_embeddings /= captions_embeddings.norm(dim=-1, keepdim=True)

            centroid_naive = captions_embeddings.mean(dim=0)
            centroid_naive /= centroid_naive.norm(dim=-1, keepdim=True).squeeze()

            zeroshot_weights_naive.append(centroid_naive)

            softmax_1 = torch.nn.functional.softmax(similarities, dim=0)
            weighted_1 = captions_embeddings * softmax_1.unsqueeze(-1)
            centroid_weighted_1 = weighted_1.sum(dim=0)
            centroid_weighted_1 /= centroid_weighted_1.norm(
                dim=-1, keepdim=True
            ).squeeze()
            zeroshot_weights_weighted_1.append(centroid_weighted_1)

            softmax_01 = torch.nn.functional.softmax(similarities / 0.1, dim=0)
            weighted_01 = captions_embeddings * softmax_01.unsqueeze(-1)
            centroid_weighted_01 = weighted_01.sum(dim=0)
            centroid_weighted_01 /= centroid_weighted_01.norm(
                dim=-1, keepdim=True
            ).squeeze()
            zeroshot_weights_weighted_01.append(centroid_weighted_01)

            softmax_001 = torch.nn.functional.softmax(similarities / 0.01, dim=0)
            weighted_001 = captions_embeddings * softmax_001.unsqueeze(-1)
            centroid_weighted_001 = weighted_001.sum(dim=0)
            centroid_weighted_001 /= centroid_weighted_001.norm(
                dim=-1, keepdim=True
            ).squeeze()
            zeroshot_weights_weighted_001.append(centroid_weighted_001)

            softmax_10 = torch.nn.functional.softmax(similarities / 10.0, dim=0)
            weighted_10 = captions_embeddings * softmax_10.unsqueeze(-1)
            centroid_weighted_10 = weighted_10.sum(dim=0)
            centroid_weighted_10 /= centroid_weighted_10.norm(
                dim=-1, keepdim=True
            ).squeeze()
            zeroshot_weights_weighted_10.append(centroid_weighted_10)

            softmax_100 = torch.nn.functional.softmax(similarities / 100.0, dim=0)
            weighted_100 = captions_embeddings * softmax_100.unsqueeze(-1)
            centroid_weighted_100 = weighted_100.sum(dim=0)
            centroid_weighted_100 /= centroid_weighted_100.norm(
                dim=-1, keepdim=True
            ).squeeze()
            zeroshot_weights_weighted_100.append(centroid_weighted_100)

        zeroshot_weights_naive = torch.stack(zeroshot_weights_naive, dim=0)
        zeroshot_weights_weighted_1 = torch.stack(zeroshot_weights_weighted_1, dim=0)
        zeroshot_weights_weighted_01 = torch.stack(zeroshot_weights_weighted_01, dim=0)
        zeroshot_weights_weighted_001 = torch.stack(
            zeroshot_weights_weighted_001, dim=0
        )
        zeroshot_weights_weighted_10 = torch.stack(zeroshot_weights_weighted_10, dim=0)
        zeroshot_weights_weighted_100 = torch.stack(
            zeroshot_weights_weighted_100, dim=0
        )

        # format key nicely, subsistuting spaces with underscores, removing commas and colons
        key_formatted = (
            key.replace(" ", "_")
            .replace(",", "")
            .replace(":", "")
            .replace("{", "")
            .replace("}", "")
            .replace(".", "")
        )

        torch.save(
            zeroshot_weights_naive,
            f"retrieved_weights/{dataset}_naive_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )
        torch.save(
            zeroshot_weights_weighted_1,
            f"retrieved_weights/{dataset}_weighted_1_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )
        torch.save(
            zeroshot_weights_weighted_01,
            f"retrieved_weights/{dataset}_weighted_01_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )
        torch.save(
            zeroshot_weights_weighted_001,
            f"retrieved_weights/{dataset}_weighted_001_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )
        torch.save(
            zeroshot_weights_weighted_10,
            f"retrieved_weights/{dataset}_weighted_10_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )
        torch.save(
            zeroshot_weights_weighted_100,
            f"retrieved_weights/{dataset}_weighted_100_{suffix}_{key_formatted}_zeroshot_weights.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)

    args = parser.parse_args()
    dataset = args.dataset
    suffix = args.suffix
    model = args.model

    main(model, dataset, suffix)
