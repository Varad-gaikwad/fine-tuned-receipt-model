# cooked.exe: Receipt Extraction with a Fine-Tuned Gemma 4

Raw receipt image in, structured JSON out.

This project fine-tunes **Gemma 4 E2B** (4-bit, LoRA) to extract four fields from scanned receipts: `company`, `date`, `address` and `total`. It compares the fine-tuned model against the plain model and against the plain model with a detailed prompt, and ships a side-by-side Gradio demo that you run yourself in a free Kaggle notebook.

Built for the HacktoberFest HackDay, Theme 3 (Fine-Tuning Gemma), Problem 3B.

```json
{
  "company": "OJC MARKETING SDN BHD",
  "date": "02/01/2019",
  "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR",
  "total": "170.00"
}
```

<!-- TODO: add a screenshot or GIF of the demo: ![demo](assets/demo.gif) -->

---

## Run it on Kaggle

There is no hosted version: the model needs an NVIDIA GPU, and keeping one running costs money. Instead, you run the demo in your own free Kaggle notebook (about 5 minutes).

**You need:** a Kaggle account with phone verification (required for GPU and Internet access).

### 1. Set up the notebook

- Create a new Kaggle notebook (or open the project notebook: `https://github.com/Varad-gaikwad/fine-tuned-receipt-model/blob/main/app.ipynb`.
- In **Settings**, set **Accelerator** to **GPU T4 x2**. Do not pick P100: Unsloth needs compute capability 7.0 or higher.
- In **Settings**, turn **Internet** on (needed for `pip`, model download and the Gradio link).

### 2. Install (Cell 1)

```bash
!pip install -q sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
!pip install -q --no-deps unsloth_zoo bitsandbytes accelerate peft trl triton unsloth
!pip install -q --no-deps transformers==5.5.0 "tokenizers>=0.22.0,<=0.23.0"
!pip install -q "huggingface_hub>=1.5.0,<2.0"
!pip install -q --no-deps --upgrade timm
!pip install -q "gradio>=6,<7" pandas scipy
```

If a later cell fails with an import or version error, restart the session (Run > Restart session) and run Cell 1 again before anything else.

### 3. Launch the demo (Cell 2)

```python
!git clone https://github.com/Varad-gaikwad/
fine-tuned-receipt-model.git
%cd 
fine-tuned-receipt-model
%run app/app.py
```

When the model finishes loading (the first run is the slowest), a public link like `https://xxxxxxxx.gradio.live` appears. Open it, upload a receipt (PNG or JPG) and click **Run Side-by-Side Extraction**. The link works only while the notebook session is running.

**Privacy:** the link is a public tunnel to your notebook. Do not upload receipts with personal or financial details.

### Options

`app/app.py` reads two optional environment variables. In a notebook, set them with `%env NAME=value` before `%run`.

---

## Results

Measured on **361 held-out test receipts** (the full test split of the Hugging Face copy of SROIE), unless stated otherwise.

### Headline

| Condition | Average accuracy, fuzzy (similarity >= 0.9) |
|---|---|
| Plain model | 72.6 |
| **Fine-tuned** | **89.8** |
| Gain | +17.2 points |

### Fine-tuned model by field (exact match)

| Field | Exact match (%) |
|---|---|
| company | 86.1 |
| date | 97.5 |
| address | 65.1 |
| total | 90.3 |
| **average** | **84.8** |

<!-- TODO: add per-field plain-model numbers (exact and fuzzy) and per-field fuzzy numbers for the fine-tuned model. -->

### Detailed-prompt baseline

The plain model with a detailed prompt was run on the **first 100 test receipts only**. It scored **70.5** average accuracy, against **88.8** for the fine-tuned model on the same 100. This subset differs from the 361-receipt headline, so do not compare 70.5 directly with 72.6.

### How to read these numbers

- **Exact match:** correct only if identical to the label after lowercasing and collapsing whitespace. It penalises harmless differences such as a missing comma.
- **Fuzzy match:** correct if `difflib.SequenceMatcher` similarity to the label is at least 0.9. It forgives formatting differences, but can also forgive small real errors such as a wrong house number.
- True accuracy lies between the two, so both are reported.

### Key findings

1. **Fine-tuning beat the plain model by 17.2 points** on average fuzzy accuracy over all 361 receipts.
2. **A detailed prompt did not close the gap:** 70.5 against 88.8 on the first 100 receipts.
3. **Address is the weakest field** (65.1 exact). Date is nearly solved (97.5).
4. **About 5 points of the average are formatting, not extraction:** the strict average is 84.8 and the fuzzy average is 89.8.
5. **Output format was never the problem.** The first evaluation run produced valid JSON 100% of the time. TODO: confirm for the final run.

### What changed since the first run

The first run (free Colab T4) used 240 training receipts and 150 test receipts. The final run (Kaggle) used **566 training receipts**, a **higher image resolution** and the **361-receipt test set**. Training data and resolution changed together, so the gain cannot be credited to either one alone. First-run numbers are superseded and not reported.

---

## Method

| Item | Value |
|---|---|
| Base model | `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit` (Gemma 4 E2B, instruction-tuned, 4-bit bitsandbytes) |
| Technique | LoRA on a 4-bit base (QLoRA-style), PEFT 0.20.0 |
| LoRA settings | rank 16, alpha 16, dropout 0, bias none; no DoRA, no rsLoRA |
| Layers adapted | Attention and MLP projections (`q/k/v/o_proj`, `gate/up/down_proj`) in the vision and language stacks, plus the per-layer input gate and projection, the vision embedding projection and the vision patch-embedder input projection |
| Trainable parameters | about 29.9 M of 5.15 B (0.58%) in the first run. TODO: confirm for the Kaggle run |
| Dataset | [`jsdnrs/ICDAR2019-SROIE`](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (Malaysian retail receipts) |
| Training data | 566 receipts. TODO: state whether this is the full train split |
| Test data | 361 receipts (full test split) |
| Image preprocessing | Long side resized to at most `TODO` px, same for training and evaluation |
| Training schedule | TODO: steps or epochs, batch size, learning rate, optimiser. (First run: 60 steps, effective batch 4, 8-bit AdamW, learning rate 2e-4, cosine) |
| Training target | JSON of the four fields |
| Hardware | Kaggle GPU: TODO (type, training time) |
| Framework | [Unsloth](https://github.com/unslothai/unsloth), TRL `SFTTrainer` |

The exact adapter settings are in [`adapter_config.json`](adapters/gemma_4_receipts_lora/adapter_config.json).

### Three evaluation conditions

1. **Plain:** base model, adapters disabled, short instruction.
2. **Plain + prompt:** base model, adapters disabled, detailed instruction describing each field and the output format.
3. **Fine-tuned:** base model with the trained LoRA adapters, short instruction.

The short instruction is:

```
Extract the company, date, address and total from this receipt. Reply with JSON only.
```

### Memory note

Decoding every full-resolution receipt into a Python list exhausted the 12.7 GB of RAM on free Colab and crashed the session. Resizing images and building only the conversations used for training fixed it. Watch for this if you retrain at higher resolution.

---

## Demo app

`app/app.py` runs the plain and fine-tuned models side by side on an uploaded receipt.

- Upload a receipt (PNG or JPG).
- Click **Run Side-by-Side Extraction**.
- Both models return JSON, shown next to each other.

Both outputs come from the same weights: the "plain" result is generated with the LoRA adapters disabled, so only one copy of the base model sits in GPU memory. Two generations run per request, so expect a few seconds per receipt. The app handles one request at a time.

---

## Repository layout

```
.
├── README.md
├── app/
│   └── app.py                      # Gradio demo
├── notebooks/
│   └── receipt_finetune.ipynb      # training, baselines, evaluation
├── results/                        # per-receipt outputs and scores per condition
├── adapters/
│   └── gemma_4_receipts_lora/      # adapter_config.json and weights (or a Hub link)
└── assets/
    ├── cooked_logo.png
    └── samples/                    # test receipts for trying the demo
```

---

## Reproduce

Use a Kaggle notebook with **GPU T4 x2** and Internet on, and the install cell from [Run it on Kaggle](#run-it-on-kaggle). Unsloth needs an NVIDIA GPU, so this does not run on Apple Silicon. Training and inference are single-GPU: the second T4 is unused.

**1. Load the model and data**

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

**2. Preprocess** (use the same function for training, baselines and evaluation)

```python
MAX_SIDE = 1024  # TODO: set to the value used in the final run
def shrink(img):
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    return img
```

**3. Run the baselines, then add LoRA and train.** Order matters: evaluate the plain model before adding adapters.

```python
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0, bias="none", random_state=3407,
)
```

The training configuration is in the notebook.

**4. Evaluate.** Per field, lowercase and collapse whitespace, then compare to the label. Run the plain conditions inside `with model.disable_adapter():`. The notebook has `evaluate_vision` and the fuzzy scorer.

**5. Load the trained adapters**

```python
from unsloth import FastVisionModel
model, processor = FastVisionModel.from_pretrained(
    "TODO-HF-USERNAME/gemma-4-receipts-lora",  # or a local path
    load_in_4bit=True,
)
FastVisionModel.for_inference(model)
```

---

## Limitations

- **One dataset, one receipt style.** Only Malaysian retail receipts from SROIE were tested. Other layouts, languages and phone photos are untested.
- **Address is unreliable.** Only 65.1% match exactly. Check addresses by hand.
- **Total is not safe for money.** At 90.3% exact match, about 1 in 10 totals is wrong. Verify before any financial use.
- **String-based scoring.** Exact match penalises formatting; fuzzy match can forgive small real errors. Neither is a human judgement.
- **Uneven prompted baseline.** It ran on 100 receipts; the other conditions ran on 361.
- **No significance tests for the final run.** The McNemar tests from the first run were not repeated.
- **Training loss is not accuracy.** Only held-out evaluation measures quality.
- **Not hosted.** It runs only in your own GPU notebook, one request at a time, and the public link ends with the session.
- **No line items.** SROIE labels only the four fields.

---

## Future work

- Vary LoRA rank, steps, learning rate and image resolution one at a time, so each change can be credited.
- Normalise currency symbols and spacing in both prediction and label before scoring.
- Spot-check failing receipts for label noise, starting with addresses.
- Test on degraded images (rotation, blur, low resolution) and real phone photos.
- Try Gemma 4 E4B and compare quality, speed and memory.
- Add line-item extraction with a dataset that has item-level labels.
- Publish a model card for the adapters on Hugging Face.

---

## Acknowledgements

- [Gemma 4](https://ai.google.dev/gemma) by Google, via [Unsloth](https://github.com/unslothai/unsloth).
- [ICDAR 2019 SROIE](https://rrc.cvc.uab.es/?ch=13), via the Hugging Face copy `jsdnrs/ICDAR2019-SROIE`.
- HacktoberFest HackDay Theme 3 starter kit.

Check the licences of the base model and the dataset before reusing the adapters or data commercially.

## Team

Varad Gaikwad ([GitHub](https://github.com/Varad-gaikwad)). TODO: add teammates and contact links if any.
