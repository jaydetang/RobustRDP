# SFT Data Generation — Multi-Task Reaction Parsing

This directory contains scripts to generate multi-task supervised fine-tuning (SFT) data for chemical reaction parsing. It includes 3 generation scripts and 1 post-processing script, producing 4 task variants that are resampled and merged into a single combined dataset.

The four task variants are:

| Task                                 | Description                                                                                           | Target Samples |
|--------------------------------------|-------------------------------------------------------------------------------------------------------|---------------:|
| **Vanilla Reaction Parsing (VRP)**   | Standard reaction parsing from the raw RxnLabel dataset with data augmentation (rotation, distortion) | 63,600 x 2     |
| **Region-Guided Reaction Parsing (RGRP)** | Parse only reactions within a specified bounding-box region in the image                        | 31,800         |
| **Prefix-Perturbed Reaction Parsing (PPRP)** | Parse reactions where some equations have perturbed/distorted bounding boxes                    | 31,800         |

---

## Directory Structure

```
sft_data_process/
├── gen_vanilla_reaction_parsing.py       # VRP: raw-label merging + augmentation
├── gen_region_guided_reaction_parsing.py # RGRP: region-guided parsing generation
├── gen_prefix_perturbed_reaction_parsing.py # PPRP: prefix-perturbed parsing generation
├── post_process.py                       # Resample & merge all task variants
├── utils/
│   ├── __init__.py
│   ├── transforms.py                     # Data augmentation transforms (RandomRotate, etc.)
│   ├── tokenizer.py                      # Tokenizer wrapper
│   └── down_sample_rxn.py               # Image resizing + JSON downsampling logic
├── raw_data/                             # Downloaded source data (see Step 1)
│   ├── images_train/                     # PNG reaction diagrams (e.g. 000000.png, ...)
│   ├── labels_train/                     # JSON annotations per image (e.g. 000000.json, ...)
│   ├── images_train_resized/             # Resized images (created by gen_vanilla_reaction_parsing.py)
│   ├── merged_train_output.json          # Merged raw labels
│   ├── train_downsampled_llm.json        # VRP real samples (~4,240 entries)
│   └── train_downsampled_rxn.json        # Intermediate format
├── aug_data/                             # Augmented data (created by gen_vanilla_reaction_parsing.py)
│   ├── images_aug_train/                 # Augmented PNG images
│   ├── images_aug_train_resized/         # Resized augmented images
│   ├── merged_aug_train_output.json      # Merged augmented labels
│   ├── train_downsampled_llm.json        # VRP augmented samples (~63,600 entries)
│   └── train_downsampled_rxn.json        # Intermediate format
├── rgrp_data/                            # Region-guided data (created by gen_region_guided_reaction_parsing.py)
│   └── train_downsampled_llm_rgrp.json   # RGRP samples
├── pprp_data/                            # Prefix-perturbed data (created by gen_prefix_perturbed_reaction_parsing.py)
│   └── train_downsampled_llm_pprp.json   # PPRP samples
└── multi_task_sft_downsampled_llm.json   # Final merged dataset (created by post_process.py)
```

---

## Step 1: Download Raw Data

Download the dataset from [Jingcz/RobustRDP-RawTrainData on Hugging Face](https://huggingface.co/datasets/Jingcz/RobustRDP-RawTrainData) and place the contents under `sft_data_process/raw_data/`.

The expected layout after download:

```
sft_data_process/raw_data/
├── images_train/              # Reaction diagram images (PNG)
└── labels_train/              # Per-image JSON annotations
```

> **Note**: Each label JSON file contains bounding-box and reaction-structure annotations for the corresponding image.

---

## Step 2: Generate Vanilla Reaction Parsing Data

Run **from the repository root**:

```bash
python sft_data_process/gen_vanilla_reaction_parsing.py
```

### What This Script Does

This script performs two major stages:

#### Stage A - Real Data Processing (gen_real_vrp_data)

1. Reads all individual label JSON files from `./sft_data_process/raw_data/labels_train/` and merges them into a single file `./sft_data_process/raw_data/merged_train_output.json`.
2. Re-indexes bounding box IDs so each image has consecutive 0-based IDs.
3. Calls `down_sample_main()` which:
   - Reads images from `./sft_data_process/raw_data/images_train/`
   - Downsamples each image (scales to fit within `target_size=1000`, ensuring width/height are multiples of 28, using LANCZOS resampling)
   - Saves resized images to `./sft_data_process/raw_data/images_train_resized/`
   - Converts annotations from the raw JSON format to the special-token format
   - Outputs `./sft_data_process/raw_data/train_downsampled_llm.json` (~4,240 real samples)
   - Also produces an intermediate `train_downsampled_rxn.json`

#### Stage B - Data Augmentation (gen_aug_vrp_data)

1. Loads the merged real data and applies 15 augmentation passes (`aug_d=15`) per sample:
   - **Rotation**: Random rotation augmentation (when `rotate_augment=True`)
   - **Distortion**: Random distortion on hue, saturation, brightness, and contrast
   - **Composite augmentation** (when `composite_augment=True`): Combines two random images into one by overlaying
2. Saves augmented images to `./sft_data_process/aug_data/images_aug_train/` and annotations to `./sft_data_process/aug_data/merged_aug_train_output.json`
3. Calls `down_sample_main()` again to produce `./sft_data_process/aug_data/train_downsampled_llm.json` (~63,600 augmented samples)          |


### Output Files (Step 2)

| Output                                   | Description                        |
|------------------------------------------|------------------------------------|
| `raw_data/merged_train_output.json`      | Merged raw labels (Stage A)        |
| `raw_data/images_train_resized/`         | Resized real images (Stage A)      |
| `raw_data/train_downsampled_llm.json`    | VRP real samples (Stage A)         |
| `aug_data/images_aug_train/`             | Augmented images (Stage B)         |
| `aug_data/merged_aug_train_output.json`  | Merged augmented labels (Stage B)  |
| `aug_data/images_aug_train_resized/`     | Resized augmented images (Stage B) |
| **`aug_data/train_downsampled_llm.json`**| **VRP augmented samples (Stage B)**|

---

## Step 3: Generate Region-Guided Reaction Parsing Data

Run **from the repository root**:

```bash
python sft_data_process/gen_region_guided_reaction_parsing.py
```

### What This Script Does

1. Reads the VRP datasets: `./sft_data_process/raw_data/train_downsampled_llm.json` (real) and `./sft_data_process/aug_data/train_downsampled_llm.json` (augmented).
2. For each image that contains **more than one** reaction:
   - Randomly selects one reaction as the anchor.
   - Defines a bounding-box region around it (expanded by a random margin of 5-20 pixels).
   - Finds all reactions whose bounding boxes are fully contained within this region.
   - Retains only regions that contain **1 or 2 reactions**.
3. Generates a new instruction with the region coordinates embedded in the prompt:
   ```
   <image> ... Given the coordinates of a region, your task is ...
   The coordinates of the given region are {x1} {y1} {x2} {y2}
   ```
4. Outputs the region-guided samples to `./sft_data_process/rgrp_data/train_downsampled_llm_rgrp.json`.

### Output Files (Step 3)

| Output                                                   | Description               |
|----------------------------------------------------------|---------------------------|
| **`rgrp_data/train_downsampled_llm_rgrp.json`**          | Region-guided samples     |

### Sample JSON Entry

```json
{
    "messages": [
        {
            "content": "<image> ... The coordinates of the given region are 120 45 380 310",
            "role": "user"
        },
        {
            "content": "<rxn><rct>140 50 180 90 <mol><cnd>190 55 240 80 <txt><prd>260 48 310 88 <mol>",
            "role": "assistant"
        }
    ],
    "images": [
        "sft_data_process/raw_data/images_train_resized/000000.png"
    ]
}
```

---

## Step 4: Generate Prefix-Perturbed Reaction Parsing Data

Run **from the repository root**:

```bash
python sft_data_process/gen_prefix_perturbed_reaction_parsing.py
```

### What This Script Does

1. Reads the same two VRP datasets as input.
2. For each image, randomly selects a subset of reactions (at least 1) to perturb.
3. For each selected reaction, applies one of three perturbation operations to ~50% of its objects (molecules/text blocks):

   | Operation | Description                                                                  |
   |-----------|------------------------------------------------------------------------------|
   | `delete`  | Removes the object entirely from the annotation                             |
   | `scale`   | Enlarges (1.5x-3.0x) or shrinks (0.2x-0.5x) the bounding box, plus a small random offset |
   | `add`     | Inserts a fake object with random bounding box coordinates and random type (`<mol>` or `<txt>`) |

4. Records which reaction indices were perturbed in a `"disturb_rxns"` field.
5. Outputs the perturbed samples to `./sft_data_process/pprp_data/train_downsampled_llm_pprp.json`.


### Output Files (Step 4)

| Output                                                   | Description               |
|----------------------------------------------------------|---------------------------|
| **`pprp_data/train_downsampled_llm_pprp.json`**          | Prefix-perturbed samples  |

### Sample JSON Entry

```json
{
    "messages": [
        {
            "content": "<image> Each image contains one or more chemical reaction flowcharts...",
            "role": "user"
        },
        {
            "content": "<rxn><rct>10 20 50 60 <mol><cnd>70 25 120 45 <txt><prd>140 22 180 62 <mol>\n<rxn><rct>200 830 560 900 <mol><cnd><prd>700 700 800 800 <txt>",
            "role": "assistant"
        }
    ],
    "images": [
        "sft_data_process/raw_data/images_train_resized/000000.png"
    ],
    "disturb_rxns": [1]
}
```

---

## Step 5: Post-Processing (Resample & Merge)

After all four generation scripts have finished, run:

```bash
python sft_data_process/post_process.py
```

### What Post-Processing Does

1. Reads the four intermediate datasets generated by the previous steps.
2. Resamples each to a target count:

   | Input File                                                              | Target Count |
   |-------------------------------------------------------------------------|:------------:|
   | `raw_data/train_downsampled_llm.json` (VRP real)                        | 63,600       |
   | `aug_data/train_downsampled_llm.json` (VRP augmented)                   | 63,600       |
   | `rgrp_data/train_downsampled_llm_rgrp.json` (RGRP)                      | 31,800       |
   | `pprp_data/train_downsampled_llm_pprp.json` (PPRP)                      | 31,800       |

   Resampling strategy:
   - **Over-sampling**: If current count < target, the data is repeated with random samples to fill the gap.
   - **Under-sampling**: If current count > target, a random subset is selected.
3. Shuffles all resampled entries together to prevent block-order bias.
4. Writes the final merged dataset to `./sft_data_process/multi_task_sft_downsampled_llm.json`.

> **Note**: The total number of samples in the final file is **63,600 + 63,600 + 31,800 + 31,800 = 191,200 entries**.

### Output Files (Step 5)

| Output                                                   | Description                        |
|----------------------------------------------------------|------------------------------------|
| **`sft_data_process/multi_task_sft_downsampled_llm.json`** | Final merged multi-task SFT dataset |

### Final Dataset Format

Each entry in the final merged dataset follows one of the message formats from the three task types. All entries contain a `"messages"` array (user instruction + assistant response) and an `"images"` array with a single image path.

**VRP entry:**
```json
{
    "messages": [
        {
            "content": "<image> Each image contains one or more chemical reaction flowcharts...",
            "role": "user"
        },
        {
            "content": "<rxn><rct>10 20 50 60 <mol><cnd>70 25 120 45 <txt><prd>140 22 180 62 <mol>",
            "role": "assistant"
        }
    ],
    "images": [
        "sft_data_process/raw_data/images_train_resized/000000.png"
    ]
}
```

**RGRP entry:**
```json
{
    "messages": [
        {
            "content": "<image> ... Given the coordinates of a region ... The coordinates of the given region are 120 45 380 310",
            "role": "user"
        },
        {
            "content": "<rxn><rct>140 50 180 90 <mol><cnd>190 55 240 80 <txt><prd>260 48 310 88 <mol>",
            "role": "assistant"
        }
    ],
    "images": [
        "sft_data_process/aug_data/images_aug_train_resized/000000.png"
    ]
}
```

**PPRP entry:**
```json
{
    "messages": [
        { "content": "<image> ...", "role": "user" },
        {
            "content": "<rxn><rct>10 20 50 60 <mol>...\n<rxn><rct>200 830 560 900 <mol>...",
            "role": "assistant"
        }
    ],
    "images": [ "sft_data_process/raw_data/images_train_resized/000000.png" ],
    "disturb_rxns": [1]
}
```

---

## Execution Summary (Full Pipeline)

Run all steps sequentially from the repository root:

```bash
# Step 1: Download raw data manually from https://huggingface.co/datasets/Jingcz/RobustRDP-RawTrainData
# into sft_data_process/raw_data/

# Step 2: Generate Vanilla Reaction Parsing data (real + augmented)
python sft_data_process/gen_vanilla_reaction_parsing.py

# Step 3: Generate Region-Guided Reaction Parsing data
python sft_data_process/gen_region_guided_reaction_parsing.py

# Step 4: Generate Prefix-Perturbed Reaction Parsing data
python sft_data_process/gen_prefix_perturbed_reaction_parsing.py

# Step 5: Resample and merge all task variants
python sft_data_process/post_process.py
```

The final output `sft_data_process/multi_task_sft_downsampled_llm.json` is ready for downstream multi-task SFT training.
