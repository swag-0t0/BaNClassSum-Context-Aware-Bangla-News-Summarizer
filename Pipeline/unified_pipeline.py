"""
Bangla News Pipeline
====================
Unified pipeline for:
  1. News Category Classification  (fine-tuned BERT)
  2. News Summarization            (fine-tuned BanglaT5)

Usage:
    pipeline = BanglaNewsPipeline()
    result   = pipeline.run("আপনার বাংলা নিউজ আর্টিকেল এখানে...")
    print(result)
"""

import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
)

# ── Memory optimisation ──────────────────────────────────────
# Disable safetensors memory-mapping to prevent Windows paging
# file errors on memory-constrained systems (<16 GB RAM).
os.environ.setdefault("SAFETENSORS_USE_MMAP", "0")

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths — anchored to project root ─────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSIFIER_PATH = os.path.join(_PROJECT_ROOT, "Models", "classifier")
SUMMARISER_PATH = os.path.join(_PROJECT_ROOT, "Models", "summarizer")


# ── Loaders ───────────────────────────────────────────────────
def load_bert_classifier():
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(
        CLASSIFIER_PATH,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_banglat5():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARISER_PATH)
    # Load in float16 to halve memory footprint (~990 MB → ~495 MB)
    model     = AutoModelForSeq2SeqLM.from_pretrained(
        SUMMARISER_PATH,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.eval()
    return model, tokenizer


# ── Pipeline ──────────────────────────────────────────────────
class BanglaNewsPipeline:
    def __init__(self):
        print("Loading classifier...")
        self.clf_model, self.clf_tokenizer = load_bert_classifier()

        print("Loading summariser...")
        self.sum_model, self.sum_tokenizer = load_banglat5()

        # Label map — update with your actual category labels
        self.id2label = self.clf_model.config.id2label

        print(f"Pipeline ready on {DEVICE}")

    @torch.no_grad()
    def classify(self, text: str) -> tuple[str, float]:
        inputs = self.clf_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(DEVICE)

        logits    = self.clf_model(**inputs).logits
        pred_id   = logits.argmax(dim=-1).item()
        category  = self.id2label[pred_id]
        confidence = torch.softmax(logits, dim=-1).max().item()
        return category, round(confidence, 4)

    @torch.no_grad()
    def summarise(self, text: str, category: str,
                  max_length: int = 128, min_length: int = 30) -> str:
        # Category-conditioned input
        input_text = f"সারসংক্ষেপ করুন [{category}]: {text}"
        inputs = self.sum_tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(DEVICE)

        gen_config = GenerationConfig(
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=False,
        )
        output_ids = self.sum_model.generate(
            **inputs,
            generation_config=gen_config,
        )
        summary = self.sum_tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return summary

    def run(self, article: str) -> dict:
        """
        Full pipeline: article → category + summary

        Args:
            article: Raw Bangla news article text

        Returns:
            dict with category, confidence, summary
        """
        category, confidence = self.classify(article)
        summary              = self.summarise(article, category)

        return {
            "category"  : category,
            "confidence": confidence,
            "summary"   : summary
        }

    def run_batch(self, articles: list[str]) -> list[dict]:
        """Run pipeline on a list of articles."""
        return [self.run(article) for article in articles]


# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = BanglaNewsPipeline()

    # Single article
    sample = (
        "ডিজিটাল নিরাপত্তা আইনে গ্রেফতার বাংলা ট্রিবিউনের খুলনা প্রতিনিধি "
        "হেদায়েত হোসেন মোল্লাকে জিজ্ঞাসাবাদের জন্য তিনদিনের রিমান্ড মঞ্জুর "
        "করেছে আদালত।"
    )

    result = pipeline.run(sample)
    print("\n── Result ──────────────────")
    print(f"Category   : {result['category']}")
    print(f"Confidence : {result['confidence']}")
    print(f"Summary    : {result['summary']}")