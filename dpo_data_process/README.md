# DPO Data Generation - Direct Preference Optimization for Reaction Parsing

This directory contains scripts to generate Direct Preference Optimization (DPO) training data for chemical reaction parsing. It includes 1 pre-processing script and 1 DPO generation script.

---

## Directory Structure

```
dpo_data_process/
├── pre_process.py              # Merge VRP data sources from sft_data_process
├── gen_dpo.py                  # Main DPO generation: inference + evaluation + filtering
├── gen_dpo.sh                  # Shell launcher for distributed DPO generation
├── dpo_data/
│   ├── train_downsampled_llm_datasource_rel_aug.json  # Merged LLM-format data (pre-process output)
│   ├── train_downsampled_rxn_datasource_rel_aug.json  # Merged RXN-format data (pre-process output)
│   └── train_downsampled_llm_dpo.json                 # Final DPO dataset with chosen/rejected pairs
└── cache_infer_results/        # Per-sample cached inference results (one JSON per index)
    ├── 0.json
    ├── 1.json
    └── ...
```

---

## Step 1: Pre-Processing

Merge the VRP real and augmented SFT data into unified files required by the DPO generation stage.

Run **from the repository root**:

```bash
python dpo_data_process/pre_process.py
```

### What Pre-Processing Does

1. **Merge LLM-format data** (`merge_llm_data`):
   - Reads `./sft_data_process/raw_data/train_downsampled_llm.json` (VRP real, ~4,240 entries)
   - Reads `./sft_data_process/aug_data/train_downsampled_llm.json` (VRP augmented, ~63,600 entries)
   - Concatenates both arrays into one unified JSON file
   - Output: `./dpo_data_process/dpo_data/train_downsampled_llm_datasource_rel_aug.json`
   - This file serves as the **`choose_file_path`** (ground-truth annotations) for DPO generation

2. **Merge RXN-format data** (`merge_rxn_data`):
   - Reads `./sft_data_process/raw_data/train_downsampled_rxn.json` (VRP real RXN-format)
   - Reads `./sft_data_process/aug_data/train_downsampled_rxn.json` (VRP augmented RXN-format)
   - Extends the `"images"` array from the first file with entries from the second
   - Output: `./dpo_data_process/dpo_data/train_downsampled_rxn_datasource_rel_aug.json`
   - This file serves as the **`eval_file_path`** (image metadata for model inference)

### Output Files

| Output | Description |
|--------|-------------|
| **`dpo_data/train_downsampled_llm_datasource_rel_aug.json`** | Merged LLM-format data (~67,840 entries) |
| **`dpo_data/train_downsampled_rxn_datasource_rel_aug.json`** | Merged RXN-format data (images array) |

---

## Step 2: Generate DPO Data

### Prerequisite: Download the Pre-trained SFT Model

The DPO generation step requires the pre-trained SFT model checkpoint. Download it from HuggingFace:

```bash
# Download the model from HuggingFace
git lfs install
git clone https://huggingface.co/Jingcz/RobustRDP-Pretrain-SFT ./eval/SFT_Model/RobustRDP-Pretrain-SFT
```

Alternatively, use `huggingface-cli`:

```bash
huggingface-cli download Jingcz/RobustRDP-Pretrain-SFT --local-dir ./eval/SFT_Model/RobustRDP-Pretrain-SFT
```

After downloading, pass the model path via `--model_path` when running the script.

Run **from the repository root**:

```bash
bash dpo_data_process/gen_dpo.sh --model_path ./eval/SFT_Model/RobustRDP-Pretrain-SFT
```


### Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | `./eval/SFT_Model/...checkpoint-47700` | Path to the pre-trained SFT model (Qwen2.5-VL). Download from [Jingcz/RobustRDP-Pretrain-SFT](https://huggingface.co/Jingcz/RobustRDP-Pretrain-SFT) on HuggingFace. |
| `--eval_file_path` | `./dpo_data_process/dpo_data/train_downsampled_rxn_datasource_rel_aug.json` | RXN-format file with image paths for evaluation |
| `--choose_file_path` | `./dpo_data_process/dpo_data/train_downsampled_llm_datasource_rel_aug.json` | LLM-format file with ground-truth annotations |
| `--dpo_output_path` | `./dpo_data_process/dpo_data/train_downsampled_llm_dpo.json` | Output path for DPO dataset |
| `--max_pixels` | 1,000,000 | Maximum pixel count for input images |
| `--max_new_tokens` | 4,500 | Maximum new tokens for model generation |
| `--infer_batch_size` | 1 | Batch size per GPU during inference |

### What DPO Generation Does

1. **Load model and processor**: Loads the pre-trained SFT model (Qwen2.5-VL) and its processor.

2. **Distributed inference**: For each image in `eval_file_path`:
   - Prepares the input with the standard reaction parsing instruction.
   - Generates predictions using the model with `max_new_tokens=4500`.
   - Cleans unwanted tokens (`<|im_end|>`, `<|im_start|>`, etc.) from the output.
   - Caches results to `./dpo_data_process/cache_infer_results/{idx}.json`.

3. **Result gathering**: Collects results from all distributed workers using `all_gather_object`. Falls back to loading cached JSON files if gathering fails.

4. **Prediction conversion**: Converts all model predictions from raw text into structured reaction annotations using `convert_pred_special_token()`.

5. **Evaluation and filtering**: For each sample:
   - Computes evaluation metrics using `ReactionEvaluator.evaluate_image()`:
     - **`overall`**: Precision, recall, and F1 across all entities.
     - **`mol_only`**: Precision, recall, and F1 for molecular entities only (with merged conditions).
   - Filters samples with **overall F1 < 0.8** (cases where the model made meaningful mistakes).

6. **DPO triplet construction**: For each selected sample:
   - **`messages`**: The user instruction (reaction parsing prompt with `<image>`).
   - **`chosen`**: The ground-truth annotation from `choose_file_path`.
   - **`rejected`**: The model's (incorrect) prediction.
   - **`images`**: Image path(s).
   - **`overall`** / **`mol_only`**: The computed metrics for debugging/analysis.

7. **Save**: Writes the DPO dataset to `--dpo_output_path`.

### DPO Dataset Format

Each entry in the DPO dataset has the following structure:

```json
{
    "messages": [
        {
            "from": "user",
            "value": "<image> \\nEach image contains one or more chemical reaction flowcharts..."
        }
    ],
    "chosen": {
        "from": "assistant",
        "value": "<rxn><rct>19 446 213 667<mol><cnd>129 299 166 451<txt><prd>..."
    },
    "rejected": {
        "from": "assistant",
        "value": "<rxn><rct>20 445 210 670<mol><cnd>133 303 171 451<txt><prd>..."
    },
    "images": [
        "sft_data_process/aug_data/images_aug_train_resized/aug_5001_3.png"
    ],
    "overall": {
        "precision": 0.625,
        "recall": 0.625,
        "f1": 0.625
    },
    "mol_only": {
        "precision": 0.75,
        "recall": 0.75,
        "f1": 0.75
    }
}
```

### Output Files

| Output | Description |
|--------|-------------|
| **`dpo_data/train_downsampled_llm_dpo.json`** | Final DPO dataset with chosen/rejected pairs |
| `cache_infer_results/*.json` | Per-sample cached inference results (intermediate, can be reused) |

---

## Execution Summary (Full Pipeline)

Run all steps from the repository root. Step 0 is a prerequisite.

```bash
# Step 0: (Prerequisite) Generate SFT data first
# python sft_data_process/gen_vanilla_reaction_parsing.py

# Step 1: Pre-process - merge VRP data sources
python dpo_data_process/pre_process.py

# Step 2: Generate DPO data (requires 8 GPUs)
bash dpo_data_process/gen_dpo.sh --model_path ./eval/SFT_Model/RobustRDP-Pretrain-SFT
```

The final output `dpo_data_process/dpo_data/train_downsampled_llm_dpo.json` is ready for downstream DPO training.

