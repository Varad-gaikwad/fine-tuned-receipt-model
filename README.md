# cooked.exe: Receipt Extraction with a Fine-Tuned Gemma 4

Raw receipt image in, structured JSON out.

This project fine-tunes **Gemma 4 E2B** (4-bit, LoRA) to extract four fields from scanned receipts: `company`, `date`, `address` and `total`. It compares the fine-tuned model against the plain model and against the plain model with a carefully written prompt, and ships a side-by-side Gradio demo.

Built for the HacktoberFest HackDay, Theme 3 (Fine-Tuning Gemma), Problem 3B.

```json
{
  "company": "OJC MARKETING SDN BHD",
  "date": "02/01/2019",
  "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR",
  "total": "170.00"
}
```

---

## Results

All numbers come from **150 held-out test receipts** (the first 150 rows of the SROIE test split). The same receipts were used for every condition.

### Exact match (strict lower bound)

| Field | Plain | Plain + detailed prompt | **Fine-tuned** | Gain (points) | p-value* |
|---|---|---|---|---|---|
| valid JSON | 100.0 | 100.0 | **100.0** | 0 | n/a |
| company | 76.0 | 78.7 | **82.7** | +6.7 | 0.052 |
| date | 88.0 | 85.3 | **94.0** | +6.0 | 0.012 |
| address | 27.3 | 33.3 | **54.7** | +27.4 | < 0.001 |
| total | 70.7 | 70.7 | **80.0** | +9.3 | 0.001 |

\*Exact McNemar test (fine-tuned vs plain), computed on exact-match results only.

### Fuzzy match (similarity >= 0.9, upper bound)

| Field | Plain | Plain + detailed prompt | **Fine-tuned** | Gain (points) |
|---|---|---|---|---|
| company | 82.0 | 86.0 | **91.3** | +9.3 |
| date | 91.3 | 88.7 | **98.0** | +6.7 |
| address | 71.3 | 70.7 | **87.3** | +16.0 |
| total | 72.0 | 72.0 | **81.3** | +9.3 |

### How to read these numbers

- **Exact match** counts a prediction as correct only if it is identical to the label after lowercasing and collapsing whitespace. It penalises harmless differences such as a missing comma.
- **Fuzzy match** counts a prediction as correct if its text similarity (`difflib.SequenceMatcher` ratio) to the label is at least 0.9. It forgives formatting differences, but it can also forgive small real errors such as a wrong house number.
- The true accuracy lies between the two. We report both.

### Key findings

1. **Fine-tuning improved every field under both metrics.** Gains on address, total and date are statistically significant on exact match. Company is borderline (p = 0.052), so treat it as suggestive.
2. **A detailed prompt did not close the gap.** The plain model with a detailed prompt stays within a few points of the plain model on every field, and is sometimes worse.
3. **The address gain is partly formatting.** The exact-match address gain (+27.4) shrinks to +16.0 under fuzzy matching, so the model partly learned the dataset's labelling conventions. The remaining gain is a real extraction improvement.
4. **Output format was never the problem.** All three conditions produced valid JSON 100% of the time.

---

### Proposed improvements

| Problem | Fix |
|---|---|
| Totals confused with other amounts | Train on more of the available receipts (only 240 were used) |
| Misread digits | Test a larger input size (for example 1280 px) |
| Currency symbols and spacing differences | Normalise both prediction and label before scoring |
| Possible label noise | Manually spot-check failing receipts |

---

## Method

| Item | Value |
|---|---|
| Base model | `unsloth/gemma-4-E2B-it`, 4-bit quantised |
| Technique | LoRA (QLoRA-style), rank 16, alpha 16, no dropout |
| Layers adapted | Vision and language layers (attention and MLP modules) |
| Trainable parameters | 29.9 M of 5.15 B (0.58%) |
| Dataset | [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (Malaysian retail receipts) |
| Training data | 240 receipts (first 240 rows of the train split) |
| Test data | 150 receipts (first 150 rows of the test split) |
| Steps | 60 steps, batch size 1, gradient accumulation 4 (effective batch 4), one epoch |
| Optimiser | 8-bit AdamW, learning rate 2e-4, cosine schedule |
| Hardware | Free Colab T4 GPU, about 6 minutes of training |
| Image preprocessing | Resized to at most 1024 px on the long side (training and evaluation) |
| Training target | JSON of the four fields |
| Framework | [Unsloth](https://github.com/unslothai/unsloth), TRL `SFTTrainer` |

### Three evaluation conditions

1. **Plain:** base model, adapters disabled, short instruction.
2. **Plain + prompt:** base model, adapters disabled, detailed instruction describing each field and the output format.
3. **Fine-tuned:** base model with the trained LoRA adapters, short instruction.

The short instruction is:

```
Extract the company, date, address and total from this receipt. Reply with JSON only.
```

### Memory notes (free Colab)

Training on all full-resolution receipts decoded into a Python list exhausted the 12.7 GB of system RAM and crashed the session. Resizing images to 1024 px and building only the conversations actually used for training kept usage at about 3.4 GB RAM and 8.4 GB VRAM.

---

## Demo app

A Gradio app runs the plain and fine-tuned models side by side on an uploaded receipt.

- Upload a receipt (PNG or JPG).
- Click **Run Side-by-Side Extraction**.
- Both models return JSON, shown next to each other.

The app runs inside the Colab session that holds the model and exposes a temporary public link (`share=True`). The link stops working when the session ends.

Extraction takes a few seconds per receipt, because two generations run per request.

---

## Repository layout

Suggested layout (adjust to match your repo):

```
.
├── README.md
├── notebooks/
│   └── receipt_finetune.ipynb      # training, baselines, evaluation
├── app/
│   └── app.py                      # Gradio demo (cooked.exe UI)
├── results/
│   ├── baseline150.csv             # plain model, per-receipt outputs and scores
│   ├── prompted150.csv             # plain + detailed prompt
│   └── finetuned150.csv            # fine-tuned model
├── adapters/
│   └── gemma_4_receipts_lora/      # trained LoRA adapters (or a link, see below)
└── assets/
    └── cooked_logo.png
```

---

## Reproduce

### 1. Environment

Use a Google Colab runtime with a **T4 GPU**. Install the versions the Gemma 4 workflow expects (from the starter kit):

```bash
pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
pip install --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
pip install --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
pip install "huggingface_hub>=1.5.0,<2.0"
pip install --no-deps --upgrade timm
pip install "gradio>=6,<7" pandas scipy
```

Unsloth needs an NVIDIA GPU, so this does not run on Apple Silicon.

### 2. Load the model and data

```python
from unsloth import FastVisionModel
from datasets import load_dataset

model, processor = FastVisionModel.from_pretrained(
    "unsloth/gemma-4-E2B-it",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

train_data = load_dataset("jsdnrs/ICDAR2019-SROIE", split="train")
test_data  = load_dataset("jsdnrs/ICDAR2019-SROIE", split="test")
```

### 3. Preprocess

```python
MAX_SIDE = 1024
def shrink(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img
```

Use the same `shrink()` for training, baselines and evaluation.

### 4. Run the baselines before adding LoRA, then train

Order matters: evaluate the plain model first, then add the adapters and train.

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
)
```

Training configuration (60 steps, effective batch 4, learning rate 2e-4, cosine schedule, `max_length=2048`) is in the notebook.

### 5. Evaluate

Per field, normalise (lowercase, collapse whitespace) and compare to the label. Run the plain conditions with `with model.disable_adapter():`. See the notebook for `evaluate_vision`, the fuzzy scorer and the McNemar test.

### 6. Use the trained adapters

```python
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(
    "path/to/gemma_4_receipts_lora", load_in_4bit=True
)
FastVisionModel.for_inference(model)
```

---

## Limitations

- **One dataset, one receipt style.** Everything is measured on 150 Malaysian retail receipts from SROIE. The results say nothing about other receipt formats, languages or phone photos.
- **Small training set.** Only 240 receipts were used, and they are the first 240 rows, not a random sample. The result is not the best this method can reach.
- **Small test set.** With 150 receipts, each receipt is about 0.7 percentage points, and the uncertainty on an accuracy near 75% is several points.
- **String-based scoring.** Exact match penalises formatting differences; fuzzy match can forgive small real errors. Neither is a human judgement of correctness.
- **Total is the weakest field.** About 1 in 5 totals is wrong. Do not use this output for anything financial without verification.
- **Significance tests cover exact match only.** Fuzzy-match significance was not tested.
- **Training loss is not accuracy.** Loss fell to about 0.05, but only the held-out evaluation measures quality.
- **Not a deployment.** The demo runs inside a Colab session, handles one request at a time, and its public link is temporary.
- **No line items.** SROIE labels only the four fields, so item, quantity and price extraction is not supported.

---

## Future work

- Train on the full training split and compare dataset sizes.
- Compare LoRA rank, step count and learning rate.
- Evaluate on degraded images (rotation, blur, low resolution) and on real phone photos.
- Try Gemma 4 E4B and compare quality, speed and memory (likely tight on a free T4).
- Add line-item extraction using a dataset that provides item-level labels.
- Publish the adapters on Hugging Face with a model card.

---

## Acknowledgements

- [Gemma 4](https://ai.google.dev/gemma) by Google, via [Unsloth](https://github.com/unslothai/unsloth).
- [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13) receipt dataset, via the Hugging Face copy `jsdnrs/ICDAR2019-SROIE`.
- HacktoberFest HackDay Theme 3 starter kit.

Check the licence and terms of the base model and of the dataset before reusing the adapters or data commercially.

## Team

Fill in: team name, members, contact links.
