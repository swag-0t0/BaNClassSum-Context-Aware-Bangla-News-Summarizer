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

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)

# ── Device ────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Paths — change if needed ──────────────────────────────────
CLASSIFIER_PATH = "./models/bert_classifier"
SUMMARISER_PATH = "./models/banglat5_summariser"


# ── Loaders ───────────────────────────────────────────────────
def load_bert_classifier():
    tokenizer = AutoTokenizer.from_pretrained(CLASSIFIER_PATH)
    model     = AutoModelForSequenceClassification.from_pretrained(CLASSIFIER_PATH).to(DEVICE)
    model.eval()
    return model, tokenizer


def load_banglat5():
    tokenizer = AutoTokenizer.from_pretrained(SUMMARISER_PATH)
    model     = AutoModelForSeq2SeqLM.from_pretrained(SUMMARISER_PATH).to(DEVICE)
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
    def classify(self, text: str) -> str:
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

        output_ids = self.sum_model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3
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