# Processed Validation Data

This directory contains the processed validation datasets for the RobustRDP project. The data is organized into two test sets: **RxnScribe test** and **RobustRDP test**.

> **Download**: The full dataset is available on Hugging Face:  
> [https://huggingface.co/datasets/Jingcz/RobustRDP-ProcessedValData](https://huggingface.co/datasets/Jingcz/RobustRDP-ProcessedValData)

---

## Directory Structure

```
processed_val_data/
│
├── RxnScribe_test/                          # RxnScribe test set
│   ├── dev_downsampled_rxn.json                     # RXN-format annotation file
│   └── images_eval_resized/                         # Resized evaluation images
│
└── RobustRDP_test/                          # RobustRDP test set
    ├── dev_downsampled_rxn.json                     # RXN-format annotation file
    └── images_eval_resized/                         # Resized evaluation images
```

---

## Subset Descriptions

### 1. RxnScribe Test (`RxnScribe_test/`)

Test set converted from the [RxnScribe](https://github.com/coleygroup/rxnscribe) benchmark. Contains real-world chemical reaction diagrams from academic literature.

| File | Description |
|------|-------------|
| `dev_downsampled_rxn.json` | RXN-format annotations with categories, bounding boxes, and reaction structures |
| `images_eval_resized/` | Evaluation images downsampled to multiples of 28 (max 1000×1000) |

**Annotation format** (in `dev_downsampled_llm.json`):

```json
{
    "messages": [
        {
            "content": "<image> \nEach image contains one or more chemical reaction flowcharts...",
            "role": "user"
        },
        {
            "content": "<rxn><rct>x1 y1 x2 y2<mol><cnd>x1 y1 x2 y2<txt><prd>x1 y1 x2 y2<mol>...",
            "role": "assistant"
        }
    ],
    "images": [
        "processed_val_data/RxnScribe_test/images_eval_resized/sample.png"
    ]
}
```

### 2. RobustRDP Test (`RobustRDP_test/`)

Test set constructed for the RobustRDP project. Contains chemical reaction diagrams with diverse layouts and reaction types.

| File | Description |
|------|-------------|
| `dev_downsampled_rxn.json` | RXN-format annotations with categories, bounding boxes, and reaction structures |
| `images_eval_resized/` | Evaluation images downsampled to multiples of 28 (max 1000×1000) |

The annotation format is identical to the RxnScribe test set (see above).

---

## Usage

After downloading from Hugging Face, place the extracted contents so that the directory structure matches the layout above. The dataset is ready to be used with the evaluation scripts in the repository root (e.g., `eval/eval.sh`).
