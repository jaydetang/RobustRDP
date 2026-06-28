import json
import math
import re
import random
import os
from PIL import Image

decouple_instruct = """
Each image contains one or more chemical reaction flowcharts, some of which may be horizontally flipped or rotated by 90°.
Given the coordinates of a region, your task is to output the reactants, conditions, and products of the equations located within that region.

Output Format:
The given region may contain multiple reactions. For each such equation, please output the bounding box coordinates and content of all reactants, conditions, and products using the following special token format:
<rxn><rct>x1 y1 x2 y2<mol>x1 y1 x2 y2<txt><cnd>x1 y1 x2 y2<txt><prd>x1 y1 x2 y2<mol>

Format specifications:
- Each reaction starts with <rxn> token
- Reactants section starts with <rct> token
- Conditions section starts with <cnd> token  
- Products section starts with <prd> token
- Each chemical entity is represented by: x1 y1 x2 y2 <type>
  - x1, y1: top-left corner coordinates (floating point numbers)
  - x2, y2: bottom-right corner coordinates (floating point numbers)
  - <type>: either <mol> for molecular structures or <txt> for text blocks
- Multiple reactions should be output sequentially using the same format
- The reactions should be organized in the order they appear in the image, from left to right and top to bottom
- If a section (reactants/conditions/products) has no entities, the section token should still be included

Example:
<rxn><rct>10 20 50 60 <mol><cnd>70 25 120 45 <txt><prd>140 22 180 62 <mol>
<rxn><rct>10 20 50 60 <mol>200 30 240 70 <mol><cnd><prd>280 28 320 68 <mol>

Input:
The coordinates of the given region are {}
"""

def get_rxn_bbox(rxn_str):
    """Parses a single rxn string to find the min/max coordinates (the bounding box)."""
    # Find all sequences of 4 numbers
    coords = re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)\s+(\d+\.?\d*)', rxn_str)
    if not coords:
        return None
    
    all_x = []
    all_y = []
    for c in coords:
        all_x.extend([float(c[0]), float(c[2])])
        all_y.extend([float(c[1]), float(c[3])])
    
    return [min(all_x), min(all_y), max(all_x), max(all_y)]

def is_inside(inner_bbox, outer_bbox):
    """Checks if inner_bbox is fully contained within outer_bbox."""
    return (inner_bbox[0] >= outer_bbox[0] and 
            inner_bbox[1] >= outer_bbox[1] and 
            inner_bbox[2] <= outer_bbox[2] and 
            inner_bbox[3] <= outer_bbox[3])


def get_image_size(image_path):
    """Get the width and height of the image"""
    try:
        with Image.open(image_path) as img:
            return img.size # (width, height)
    except Exception as e:
        print(f"Unable to read image {image_path}: {e}")
        return None, None
    

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    
    for entry in data:
        img_w, img_h = get_image_size(entry["images"][0])

        assistant_content = entry["messages"][1]["content"]
        
        # 1. Split into individual reactions
        rxns = re.findall(r'(<rxn>.*?)(?=<rxn>|$)', assistant_content, re.DOTALL)
        
        # 2. Skip if only one or zero reactions
        if len(rxns) <= 1:
            continue

        # Pre-calculate bboxes for all reactions in this image
        rxn_info = []
        for r in rxns:
            bbox = get_rxn_bbox(r)
            if bbox:
                rxn_info.append({"content": r.strip(), "bbox": bbox})

        if not rxn_info:
            continue

        # Try up to 10 random selections
        attempts = 0
        tried_indices = set()
        
        while attempts < 10 and len(tried_indices) < len(rxn_info):
            idx = random.randint(0, len(rxn_info) - 1)
            if idx in tried_indices:
                attempts += 1
                continue
            tried_indices.add(idx)
            
            # 3. Define the region based on a randomly selected reaction
            target_bbox = rxn_info[idx]["bbox"]
            # Expand by 20 units
            x1 = max(0, int(math.floor(target_bbox[0] - random.randint(5, 20))))
            y1 = max(0, int(math.floor(target_bbox[1] - random.randint(5, 20))))
            x2 = min(img_w, int(math.ceil(target_bbox[2] + random.randint(5, 20))))
            y2 = min(img_h, int(math.ceil(target_bbox[3] + random.randint(5, 20))))
            
            region = [x1, y1, x2, y2]
            
            # 4. Find all reactions fully contained within this region
            matched_rxns = []
            for info in rxn_info:
                if is_inside(info["bbox"], region):
                    matched_rxns.append(info["content"])
            
            # 5. Requirement: 1 to 2 reactions allowed in the region
            # Due to bidirectional arrows, there may be two equations in the region
            if 1 <= len(matched_rxns) <= 2:
                # Construct the sample
                chosen_region_str = f"{region[0]} {region[1]} {region[2]} {region[3]}"
                new_entry = {
                    "messages": [
                        {
                            "content": f"<image> {decouple_instruct.format(chosen_region_str)}",
                            "role": "user"
                        },
                        {
                            "content": "\n".join(matched_rxns),
                            "role": "assistant"
                        }
                    ],
                    "images": entry["images"]
                }
                new_data.append(new_entry)
                break
            else:
                attempts += 1
        
    return new_data



if __name__ == '__main__':
    # Generate Region-Guided Reaction Parsing SFT data from 4240 real samples and 63600 aug. samples
    files = ["./sft_data_process/raw_data/train_downsampled_llm.json", "./sft_data_process/aug_data/train_downsampled_llm.json"]

    res = []
    for file_name in files:
        if os.path.exists(file_name):
            processed_data = process_json(file_name)
            res.extend(processed_data)

    output_name = "./sft_data_process/rgrp_data/train_downsampled_llm_rgrp.json"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    print(f"File processing completed, saved to {output_name}, number of valid samples: {len(res)}")