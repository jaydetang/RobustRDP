# Validation Data Processing - Generate Processed Validation Data

This directory contains scripts to process raw validation data into the format required by RobustRDP evaluation. It includes 2 processing scripts, each handling a different test set.

---

## Directory Structure

```
raw_val_data/
├── README.md                                   # This file
├── gen_processed_val_data_rxnscribe_test.py     # Process RxnScribe test data
├── gen_processed_val_data_robustrdp_test.py     # Process RobustRDP test data
├── RxnScribe_test/
│   ├── dev.json                                 # Raw RxnScribe annotation file
│   └── images_eval/                             # Raw RxnScribe images
├── RobustRDP_test/
│   ├── labels_eval/                             # Raw RobustRDP label files (one JSON per image)
│   ├── images_eval/                             # Raw RobustRDP images
└── utils/
    └── down_sample_rxn.py                       # Shared utility: image downsampling & format conversion
```

---

### Prerequisite: Download the Raw Validation Data

The raw validation data is hosted on HuggingFace. Download it before running the processing scripts:

```bash
# Download the dataset from HuggingFace
git lfs install
git clone https://huggingface.co/datasets/Jingcz/RobustRDP-RawValData ./raw_val_data
```

Alternatively, use `huggingface-cli`:

```bash
huggingface-cli download Jingcz/RobustRDP-RawValData --local-dir ./raw_val_data --repo-type dataset
```

After downloading, the `raw_val_data/` directory will contain the raw data (as shown in the Directory Structure above).

---

## Step 1: Process RxnScribe Test Data

Converts the RxnScribe test set into the RobustRDP RXN-format.

Run **from the repository root**:

```bash
python raw_val_data/gen_processed_val_data_rxnscribe_test.py
```

### What This Script Does

1. **Reads raw data**:
   - Annotation file: `./raw_val_data/RxnScribe_test/dev.json`
   - Images directory: `./raw_val_data/RxnScribe_test/images_eval`

2. **Downsamples images** (`down_sample_main` with `mode="dev"`):
   - Resizes all images so that both width and height are multiples of 28 and the total pixel count does not exceed 1000×1000 (1,000,000 pixels).
   - Saves the downsampled images to a new directory with the `_resized` suffix.

3. **Converts annotations to RXN format**:
   - Scales bounding box coordinates according to the image resize factor.
   - Converts annotation format into the standard RobustRDP RXN-format JSON (`dev_downsampled_rxn.json`), which contains categories, image metadata, and per-image bounding boxes with reactions.
   - Also generates a corresponding LLM-format file (`dev_downsampled_llm.json`) with messages in the reaction parsing instruction format.

### Output Files

| Output | Description |
|--------|-------------|
| **`raw_val_data/RxnScribe_test/dev_downsampled_rxn.json`** | Processed RXN-format annotations for RxnScribe test set |
| `raw_val_data/RxnScribe_test/images_eval_resized/` | Downsampled images (input for model evaluation) |

---

## Step 2: Process RobustRDP Test Data

Converts the RobustRDP test set into the RobustRDP RXN-format.

Run **from the repository root**:

```bash
python raw_val_data/gen_processed_val_data_robustrdp_test.py
```

### What This Script Does

1. **Merge individual label files** (`gen_processed_val_data`):
   - Reads all JSON files from `./raw_val_data/RobustRDP_test/labels_eval/`.
   - Merges them into a single JSON structure with standard category definitions (structure, text, identifier, supplement).
   - Re-indexes bounding box IDs to be consecutive integers from 0 to n-1, updating references in reactions accordingly.
   - Saves the merged result as an intermediate file: `./raw_val_data/RobustRDP_test/merged_eval_output.json`.

2. **Downsamples images** (`down_sample_main` with `mode="dev"`):
   - Resizes all images so that both width and height are multiples of 28 and the total pixel count does not exceed 1000×1000.
   - Saves the downsampled images to a new directory with the `_resized` suffix.

3. **Converts annotations to RXN format**:
   - Scales bounding box coordinates according to the image resize factor.
   - Converts annotation format into the standard RobustRDP RXN-format JSON (`dev_downsampled_rxn.json`).
   - Also generates a corresponding LLM-format file (`dev_downsampled_llm.json`).

### Output Files

| Output | Description |
|--------|-------------|
| **`raw_val_data/RobustRDP_test/dev_downsampled_rxn.json`** | Processed RXN-format annotations for RobustRDP test set |
| `raw_val_data/RobustRDP_test/images_eval_resized/` | Downsampled images (input for model evaluation) |

---

## Execution Summary (Full Pipeline)

Run both steps from the repository root in order:

```bash
# Step 1: Process RxnScribe test data
python raw_val_data/gen_processed_val_data_rxnscribe_test.py

# Step 2: Process RobustRDP test data
python raw_val_data/gen_processed_val_data_robustrdp_test.py
```

After running both scripts, the `raw_val_data/` directory will contain the two processed test sets ready for model evaluation.

