# BaNClassSum: Context-Aware Bangla News Summarizer

![UI screenshot](assets/UI.png)

A research repository for Bangla news category-aware summarization. The project combines fine-tuned BERT classification with a BanglaT5 summarization model to generate summaries conditioned on predicted news categories.

## Repository Structure

- `Data/`
  - `raw/` � raw news dataset files, including `ittefaq.jsonl`
  - `extractive_summary/` � extracted summaries and related dataset artifacts such as `all_summary.jsonl`
- `Models/` � trained model artifacts and checkpoints
- `Notebooks/` � exploratory notebooks for training, fine-tuning, and extractive summarization experiments
- `Pipeline/` � reusable pipeline implementation in `unified_pipeline.py`
- `evaluation/` � evaluation utilities and results data

## Requirements

This repository uses the Hugging Face Transformers library and PyTorch.

Recommended environment:

```bash
python -m pip install --upgrade pip
python -m pip install torch transformers
```

If you are using a GPU, install the appropriate `torch` package for your CUDA version.

## Pipeline Usage

The main reusable component is `Pipeline/unified_pipeline.py`.
It loads a classifier and summarizer from local model directories and provides a `BanglaNewsPipeline` class.

### Expected model paths

By default, the pipeline loads models from:

- `./models/bert_classifier`
- `./models/banglat5_summariser`

Update these paths in `Pipeline/unified_pipeline.py` if your model directories differ.

### Example

```python
from Pipeline.unified_pipeline import BanglaNewsPipeline

pipeline = BanglaNewsPipeline()
article = "????? ????? ???? ???????? ?????..."
result = pipeline.run(article)
print(result)
```

The returned result contains:

- `category` � predicted news category
- `confidence` � classifier confidence score
- `summary` � generated Bangla summary

## Notes

- `BanglaNewsPipeline.classify` uses the classifier model's `id2label` mapping for category names.
- `BanglaNewsPipeline.summarise` constructs a category-conditioned prompt before generating the summary.
- The pipeline uses CPU by default and will switch to GPU automatically if available.

## Notebooks

Key notebooks include:

- `Notebooks/extractive-summarisation-bert-tfidf-pos-lime.ipynb`
- `Notebooks/fine-tune-bert-in-newsclass-refined.ipynb`

These notebooks contain experiments for extractive summarization and classifier fine-tuning.

## Dataset

The dataset is organized under `Data/`.
`Data/raw/ittefaq.jsonl` appears to be a raw Bangla news corpus.
`Data/extractive_summary/all_summary.jsonl` contains extractive summary artifacts used for evaluation or preprocessing.

## License

Add a license file or license section here if needed.
