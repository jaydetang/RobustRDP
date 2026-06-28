# Pretraining Data Generation - LayoutDrivenSynthesizer

This directory contains scripts to synthesize chemical reaction diagram data for pretraining. It includes 4 layout generation scripts and 1 post-processing script.

---

## Directory Structure

```
pretrain_data_process/
├── gen_single_line.py       # Generate single-line chain-style reactions
├── gen_multi_line.py        # Generate multi-line chain-style reactions
├── gen_branch.py            # Generate branching reactions
├── gen_cycle.py             # Generate cyclic reactions
├── post_process.py          # Resize images, convert annotations, merge all types
├── utils.py                 # Shared utilities (Indigo rendering, fonts, etc.)
├── raw_data/                # Source data
│   ├── pubchem/train_1m.csv # Molecule SMILES library
│   ├── conditions.txt       # Reaction condition text pool
│   └── ttf-ms-win10/        # Font files for rendering
├── images/                  # Generated output (created by gen_*.py)
│   ├── single_line/
│   ├── multi_line/
│   ├── branch/
│   └── cycle/
└── images_pretrain_resized/ # Post-processed output (created by post_process.py)
    ├── single_line_resized/
    ├── multi_line_resized/
    ├── branch_resized/
    ├── cycle_resized/
    └── pretrain_downsampled_llm_6w.json   # Merged final dataset
```

---

## Step 1: Generate Raw Data

Run the four generation scripts **from the repository root**:

```bash
# Single-line chain-style reactions
python pretrain_data_process/gen_single_line.py

# Multi-line chain-style reactions
python pretrain_data_process/gen_multi_line.py

# Branching reactions
python pretrain_data_process/gen_branch.py

# Cyclic reactions
python pretrain_data_process/gen_cycle.py
```

### Configuration Per Script

| Script            | `IMAGE_NUM` | `OUTPUT_DIR` (relative to repo root)        | `PREFIX`        | `max_workers` |
|-------------------|-------------|----------------------------------------------|------------------|---------------|
| `gen_single_line` | 10          | `./pretrain_data_process/images/single_line` | `single_line_{}` | 120           |
| `gen_multi_line`  | 10          | `./pretrain_data_process/images/multi_line`  | `multi_line_{}`  | 120           |
| `gen_branch`      | 20          | `./pretrain_data_process/images/branch`      | `branch_{}`      | 120           |
| `gen_cycle`       | 20          | `./pretrain_data_process/images/cycle`       | `cycle_{}`       | 120           |

> **Note**: `IMAGE_NUM` is currently set to small values (10 or 20) for testing. In the paper, the following settings were used:
> - `gen_single_line.py`: `IMAGE_NUM = 10000`
> - `gen_multi_line.py`: `IMAGE_NUM = 10000`
> - `gen_branch.py`: `IMAGE_NUM = 20000`
> - `gen_cycle.py`: `IMAGE_NUM = 20000`
>
> **Adjust `IMAGE_NUM` in each script to your desired scale before production runs.**

### Generation Details

Each script:

1. Reads molecular SMILES from `./pretrain_data_process/raw_data/pubchem/train_1m.csv`
2. Reads reaction conditions from `./pretrain_data_process/raw_data/conditions.txt`
3. Uses [Indigo](https://lifescience.opensource.epam.com/indigo/) to render molecule images
4. Lays out reactions on a 2000×2000 white canvas with random positions, arrows, and condition labels
5. Produces a structured JSON annotation with bounding box coordinates (normalized to 0–1000) and content types (`"molecule"` or `"text"`)
6. Saves images as PNG and the annotation array as a single JSON file

### Output Per Script

For example, `gen_single_line.py` produces:

| Output             | Path                                               |
|--------------------|----------------------------------------------------|
| Images             | `./pretrain_data_process/images/single_line/*.png` |
| Annotations (JSON) | `./pretrain_data_process/images/single_line.json`  |

The annotation JSON is an array of entries, each with this structure:

```json
[
    {
        "messages": [
            {
                "content": "<image>\\nThe chemical reaction flowchart ...",
                "role": "user"
            },
            {
                "content": "<answer> [ { \"reactants\": [...], \"conditions\": [...], \"products\": [...] } ] </answer>",
                "role": "assistant"
            }
        ],
        "images": [
            "./pretrain_data_process/images/single_line/single_line_0.png"
        ]
    },
    ...
]
```

---

## Step 2: Post-Processing

After all four generation scripts have finished, run:

```bash
python pretrain_data_process/post_process.py
```

### What Post-Processing Does

1. **Processes all 4 types** (`single_line`, `multi_line`, `branch`, `cycle`) in sequence.
2. For each type:
   - Loads annotations from `./pretrain_data_process/images/{type}.json`
   - Replaces the user message with a unified special-token instruction
   - Downsamples each image: scales to fit within `target_size=1000` while keeping width/height as multiples of 28, using LANCZOS resampling
   - Moves any molecules mistakenly labeled as `conditions` into `reactants`
   - Converts the `<answer> JSON </answer>` annotation into the special token format
   - Updates the image paths to point to the resized images
   - Saves resized images to `./pretrain_data_process/images_pretrain_resized/{type}_resized/` and annotations to `{type}_resized.json`
3. **Merges all 4 types** into one combined file.

### Configuration

| Parameter     | Value |
|---------------|-------|
| `target_size` | 1000  |
| `num_workers` | 64    |

### Special Token Format

Each reaction is converted from JSON to a compact token sequence:

```
<rxn><rct>x1 y1 x2 y2<mol><cnd>x1 y1 x2 y2<txt><prd>x1 y1 x2 y2<mol>
```

Token meanings:

| Token   | Meaning                          |
|---------|----------------------------------|
| `<rxn>` | Start of one reaction            |
| `<rct>` | Reactants section                |
| `<cnd>` | Conditions section               |
| `<prd>` | Products section                 |
| `<mol>` | Molecule entity type             |
| `<txt>` | Text block entity type           |

Coordinates are de-normalized from 0–1000 back to original pixel space, scaled by the resize factor, clamped to `[0, new_width/height]`, and rounded to integers.

### Output Files

| Output                                                    | Description                        |
|-----------------------------------------------------------|------------------------------------|
| `images_pretrain_resized/single_line_resized/` + `.json`  | Resized single-line data           |
| `images_pretrain_resized/multi_line_resized/` + `.json`   | Resized multi-line data            |
| `images_pretrain_resized/branch_resized/` + `.json`       | Resized branch data                |
| `images_pretrain_resized/cycle_resized/` + `.json`        | Resized cycle data                 |
| **`images_pretrain_resized/pretrain_downsampled_llm_6w.json`** | **Merged dataset (all 4 types)** |

### Merged Dataset Format

The final file `pretrain_downsampled_llm_6w.json` contains the entries from all 4 types concatenated. Each entry has this final format:

```json
{
    "messages": [
        {
            "content": "<image> \\nEach image contains ...",
            "role": "user"
        },
        {
            "content": "<rxn><rct>10 20 50 60<mol><cnd>70 25 120 45<txt><prd>140 22 180 62<mol>\\n<rxn>...",
            "role": "assistant"
        }
    ],
    "images": [
        "pretrain_data_process/images_pretrain_resized/single_line_resized/single_line_0.png"
    ]
}
```

