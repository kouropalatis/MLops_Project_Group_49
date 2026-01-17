## Technical Summary

**Dataset**
- Source: Financial Phrase Bank v1.0 (local files only).
- Files: `Sentences_AllAgree.txt`, `Sentences_75Agree.txt`, `Sentences_66Agree.txt`, `Sentences_50Agree.txt`.
- Parsing: Each line is `sentence@sentiment` with `sentiment ∈ {negative, neutral, positive}`.
- Labels: Mapped to integers `{negative: 0, neutral: 1, positive: 2}`.
- Tokenization: Lowercase whitespace split; vocabulary built with special tokens `PAD=0`, `UNK=1`.
- Collation: Batches are padded to max length in batch, producing `inputs: [B, T]`, `targets: [B]`.

**Model**
- Architecture: Bag-of-words classifier via average embeddings.
- Components: `Embedding(vocab_size, D, padding_idx=0)` → mask non-pad → average → `Linear(D, 3)`.
- Output: Logits `[B, 3]` for 3-way sentiment.

**Training**
- Loss: Cross-Entropy between logits and label ids.
- Optimizer: Adam.
- DataLoader: Tunable performance knobs: `num_workers`, `pin_memory`, `persistent_workers`, `prefetch_factor`.
- Metrics: Epoch loss and accuracy.

**Usage**
```powershell
# Set dataset path
set PHRASEBANK_PATH=F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0

# Train
python src\project\train.py

# Preprocess cache (optional)
python -m project.data preprocess "F:\Business Analytics Dk\MLOps\FinancialPhraseBank-v1.0" "data\processed" --agreement AllAgree
```

**Notes**
- Licensing: CC BY-NC-SA 3.0 — non‑commercial use with attribution.
- Agreement choice affects label quality; default is `AllAgree`.
