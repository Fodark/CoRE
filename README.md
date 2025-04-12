<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2411.00988-b31b1b.svg)](https://arxiv.org/abs/2411.00988)

# Retrieval-enriched zero-shot image classification in low-resource domains

## EMNLP 2024 (Main)

[Nicola Dall'Asen](https://scholar.google.com/citations?user=e7lgiYYAAAAJ), [Yiming Wang](https://scholar.google.com/citations?user=KBZ3zrEAAAAJ), [Enrico Fini](https://scholar.google.com/citations?user=OQMtSKIAAAAJ), [Elisa Ricci](https://scholar.google.com/citations?user=xf1T870AAAAJ)

______________________________________________________________________

![](static/images/method.png)

</div>
Low-resource domains, characterized by scarce data and annotations, present significant challenges for language and visual understanding tasks, with the latter much under-explored in the literature. Recent advancements in Vision-Language Models (VLM) have shown promising results in high-resource domains but fall short in low-resource concepts that are under-represented (e.g. only a handful of images per category) in the pre-training set. We tackle the challenging task of zero-shot low-resource image classification from a novel perspective. By leveraging a retrieval-based strategy, we achieve this in a training-free fashion. Specifically, our method, named CoRE (Combination of Retrieval Enrichment), enriches the representation of both query images and class prototypes by retrieving relevant textual information from large web-crawled databases. This retrieval-based enrichment significantly boosts classification performance by incorporating the broader contextual information relevant to the specific class. We validate our method on a newly established benchmark covering diverse low-resource domains, including medical imaging, rare plants, and circuits. Our experiments demonstrate that CORE outperforms existing state-of-the-art methods that rely on synthetic data generation and model fine-tuning.

______________________________________________________________________

## Env setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

or with `uv`

```bash
uv sync
source .venv/bin/activate
```

## Datasets

Download the relevant datasets from the following links.

- [Circuits](https://github.com/xiaobai1217/Low-Resource-Vision)
- [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
- [iNaturalist](https://github.com/visipedia/inat_comp/blob/master/2021/README.md)

### For database construction

- [CC12M webdataset](https://huggingface.co/datasets/pixparse/cc12m-wds)
- [COYO700M](https://huggingface.co/datasets/kakaobrain/coyo-700m)

## Preprocess datasets

To build the textual databases to compute image-to-text and text-to-text retrieval, look at the following files.

### SigLIP for image-to-text retrieval

For CC12M look at [CC12M preprocessing SigLIP](src/preprocessing/cc12m_siglip.py)

For COYO700M look at [COYO preprocessing SigLIP](src/preprocessing/coyo_siglip.py)

### Mistral for text-to-text retrieval

For CC12M look at [CC12M preprocessing Mistral](src/preprocessing/cc12m_mistral.py)

For COYO700M look at [COYO preprocessing Mistral](src/preprocessing/coyo_mistral.py)

Then look at the awesome guide by (Alessandro Conti)\[https://github.com/altndrr/vic/issues/8#issuecomment-1594732756\] to create the fast FAISS index with the embeddings.

### Already-computed databases

We will shortly release our pre-computed databases for the two datasets above.

## CoRE

#### Custom datasets

You just need to create 3 csv files like those you can find in [the artifacts folder](artifacts/data/), specifying the index to class name, the "train" samples and the test samples.

### Image-to-text retrieval

Run [Image-to-text retrieval](src/core/core_img2txt_retrieval.py) specifying the dataset, the split of the dataset, and which database you want to use (e.g. `coyo_siglip` or `cc12m_siglip`). The resulting embeddings will be stored in `retrieved_embeddings`.

### Text-to-text retrieval

First run [Text-to-text retrieval](src/core/core_txt2txt_retrieval.py), specifyint the dataset, the database to use, and the embedding model. If you want to specify a custom dataset with custom retrieval prompt, modify `dataset_data` in this file. The resulting embeddings will be stored in `results`.

Once you obtain the embeddings, you can build the retrieved zero-shot weights with different temperatures by running [the create zero-shot script](src/core/core_zeroshot_from_retrieved.py) by specifying which model has been used (siglip/mistral), the dataset, and the suffix indicating whether you used the `common` object names (for Circuits and HAM10000) or the `premerged` (merged common/scientific names) for iNaturalist.

### Enriched zero-shot predictions

Lastly, you can use the image-to-text retrieval and the text-to-text retrieval to run the enriched zero-shot predictions using [this script](src/core/core.py).

You need to specify: the dataset, the database used for image-to-text retrieval (cc12m/coyo), the dataset split (train/test), the alpha and beta values to test for the merging (can be floats or lists) and the temperature for the image-to-text retrieval weight distribution computation.

For the text-to-text retrieval part, look at line 17 onwards to load your saved retrieved zero-shot weights, you have to specify the temperature used to extract those (WIP: infer from the file name).

The output will be the parameters that led to the best Acc@1, if you run the script on the train split then you can use the output parameters on your test split.

______________________________________________________________________

If you find our research useful, please cite us as:

```
@inproceedings{dallasen2024retrieval,
    title = "Retrieval-enriched zero-shot image classification in low-resource domains",
    author = "Dall{'}Asen, Nicola  and
      Wang, Yiming  and
      Fini, Enrico  and
      Ricci, Elisa",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-main.1186/",
    doi = "10.18653/v1/2024.emnlp-main.1186",
}
```

# Website License

<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.
