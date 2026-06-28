import json
import random
import os

def resample_data(file_path, target_count):
    """
    Resample a single file:
    - If data count > target_count, random sampling without replacement (under-sampling)
    - If data count < target_count, cycle fill and pad (over-sampling)
    """
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist, skipping.")
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    current_count = len(data)
    print(f"Reading file: {file_path} | Current count: {current_count} | Target count: {target_count}")

    if current_count >= target_count:
        # Under-sampling: randomly select target_count samples
        resampled = random.sample(data, target_count)
    else:
        # Over-sampling: take all, then randomly fill the remainder
        full_cycles = target_count // current_count
        remainder = target_count % current_count
        resampled = data * full_cycles + random.sample(data, remainder)
    
    return resampled

def main():
    # Configure your file paths and target counts
    config = [
        {"path": "./sft_data_process/raw_data/train_downsampled_llm.json", "target": 63600},
        {"path": "./sft_data_process/aug_data/train_downsampled_llm.json", "target": 63600},
        {"path": "./sft_data_process/rgrp_data/train_downsampled_llm_rgrp.json", "target": 31800},
        {"path": "./sft_data_process/pprp_data/train_downsampled_llm_pprp.json", "target": 31800}
    ]

    final_combined_data = []

    for item in config:
        sampled_part = resample_data(item["path"], item["target"])
        final_combined_data.extend(sampled_part)

    # Shuffle the total data order to prevent model from learning block-wise bias
    print("Globally shuffling data...")
    random.shuffle(final_combined_data)

    output_path = "./sft_data_process/multi_task_sft_downsampled_llm.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_combined_data, f, ensure_ascii=False, indent=4)
    
    print(f"--- Processing completed ---")
    print(f"Total samples: {len(final_combined_data)}")
    print(f"Save path: {output_path}")

if __name__ == "__main__":
    main()