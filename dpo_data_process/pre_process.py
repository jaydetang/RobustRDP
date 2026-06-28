import json
import random
import os


def merge_llm_data(input_paths, output_path):
    """
    Merge multiple LLM JSON data files into one.

    Args:
        input_paths (list): List of paths to JSON files to merge.
        output_path (str): Path to save the merged JSON output.
    """
    final_data = []
    for file_path in input_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        final_data.extend(data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, ensure_ascii=False, indent=4)

    print(f"--- Processing complete: LLM data merged to {output_path} ---")


def merge_rxn_data(input_paths, output_path):
    """
    Merge multiple RXN JSON data files, prefixing image file names with their
    respective source directories.

    Args:
        input_paths (list): List of paths to RXN JSON files to merge.
        image_dirs (list): List of image directories corresponding to each input file.
        output_path (str): Path to save the merged JSON output.
    """
    with open(input_paths[0], 'r', encoding='utf-8') as f:
        data0 = json.load(f)

    with open(input_paths[1], 'r', encoding='utf-8') as f:
        data1 = json.load(f)

    data0["images"].extend(data1["images"])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data0, f, ensure_ascii=False, indent=4)

    print(f"--- Processing complete: RXN data merged to {output_path} ---")


if __name__ == "__main__":
    merge_llm_data(
        input_paths=[
            "./sft_data_process/raw_data/train_downsampled_llm.json",
            "./sft_data_process/aug_data/train_downsampled_llm.json",
        ],
        output_path="./dpo_data_process/dpo_data/train_downsampled_llm_datasource_rel_aug.json",
    )

    merge_rxn_data(
        input_paths=[
            "./sft_data_process/raw_data/train_downsampled_rxn.json",
            "./sft_data_process/aug_data/train_downsampled_rxn.json",
        ],
        output_path="./dpo_data_process/dpo_data/train_downsampled_rxn_datasource_rel_aug.json",
    )
