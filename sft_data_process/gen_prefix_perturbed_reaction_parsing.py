import json
import re
import random
import os

reaction_parsing_instruct = """
Each image contains one or more chemical reaction flowcharts, some of which may be horizontally flipped or rotated by 90°.
Your task is to identify and extract the structured information of each reaction, including reactants, products, and reaction conditions.

Output Format:
The chemical reaction flowchart contains multiple reaction equations. Please output the bounding box coordinates and content of all reactants, conditions, and products in each equation using the following special token format:
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

Example for multiple reactions:
<rxn><rct>10 20 50 60 <mol><cnd>70 25 120 45 <txt><prd>140 22 180 62 <mol>
<rxn><rct>200 30 240 70 <mol><cnd><prd>280 28 320 68 <mol>
"""


# Define the regex for matching coordinates and types
OBJ_PATTERN = r'(\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*)\s*(<mol>|<txt>)'

def perturb_coords(coord_str):
    """Perturb a coordinate string by enlarging or shrinking it."""
    try:
        nums = [float(x) for x in coord_str.split()]
        scale = random.uniform(1.5, 3.0) if random.random() > 0.5 else random.uniform(0.2, 0.5)
        # Apply a large-scale offset to x2 and y2
        w = nums[2] - nums[0]
        h = nums[3] - nums[1]
        nums[0] = max(0, nums[0] + random.randint(-20, 20))
        nums[1] = max(0, nums[1] + random.randint(-20, 20))
        nums[2] = nums[0] + w * scale
        nums[3] = nums[1] + h * scale

        return " ".join([f"{int(x)}" for x in nums])
    except:
        return coord_str

def process_rxn_string(rxn_str):
    """Parse a single <rxn> string into a structured dictionary."""
    parts = {}
    for tag in ['rct', 'cnd', 'prd']:
        # Use regex to extract the content of each tag
        pattern = f'<{tag}>(.*?)(?=<rct>|<cnd>|<prd>|$)'
        match = re.search(pattern, rxn_str, re.DOTALL)
        content = match.group(1) if match else ""
        # Extract all objects in this section (coordinates + type)
        objs = [{"coords": m[0], "type": m[1]} for m in re.findall(OBJ_PATTERN, content)]
        parts[tag] = objs
    return parts

def render_rxn_string(struct_rxn):
    """Convert a structured dictionary back into a string."""
    res = "<rxn>"
    for tag in ['rct', 'cnd', 'prd']:
        res += f"<{tag}>"
        for obj in struct_rxn[tag]:
            res += f"{obj['coords']}{obj['type']}"
    return res

def process_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_data = []
    for entry in data:
        assistant_content = entry["messages"][1].get("content", "")
        # 1. Split equations
        rxn_strings = re.findall(r'(<rxn>.*?)(?=<rxn>|$)', assistant_content, re.DOTALL)
        num_rxns = len(rxn_strings)
        
        # 2. Determine the number of perturbed equations: 40%
        num_to_perturb = int(num_rxns * 0.4)
        if num_to_perturb < 1:
            continue
            
        # Randomly select the indices of equations to perturb
        perturb_indices = random.sample(range(num_rxns), num_to_perturb)
        final_rxns = []
        disturbed_rxn_ids = []

        for i, rxn_str in enumerate(rxn_strings):
            if i not in perturb_indices:
                final_rxns.append(rxn_str.strip())
                continue
            
            # Record the indices of the original equations that were perturbed
            disturbed_rxn_ids.append(i)
            
            # Parse the structure
            struct = process_rxn_string(rxn_str)
            
            # Collect all objects in the equation and perturb 50% of them
            all_objs_ref = []
            for tag in ['rct', 'cnd', 'prd']:
                for obj_idx, obj in enumerate(struct[tag]):
                    all_objs_ref.append((tag, obj_idx))
            
            if not all_objs_ref:
                final_rxns.append(rxn_str.strip())
                continue

            num_obj_to_perturb = max(1, int(len(all_objs_ref) * 0.5))
            target_objs = random.sample(all_objs_ref, num_obj_to_perturb)
            
            # Apply perturbation operations
            # To avoid index misalignment during deletion, mark first and process later
            to_remove = []
            for tag, obj_idx in target_objs:
                op = random.choice(['delete', 'scale', 'add'])
                
                if op == 'delete':
                    to_remove.append((tag, obj_idx))
                elif op == 'scale':
                    struct[tag][obj_idx]['coords'] = perturb_coords(struct[tag][obj_idx]['coords'])
                elif op == 'add':
                    # Add a distracting object near the current position
                    rand_x0, rand_y0 = random.randint(0,999), random.randint(0,999)
                    rand_x1, rand_y1 = min(999, rand_x0+random.randint(10,100)), min(999, rand_y0+random.randint(10,100))

                    fake_obj = {
                        "coords": f"{rand_x0} {rand_y0} {rand_x1} {rand_y1}",
                        "type": random.choice(['<mol>', '<txt>'])
                    }
                    struct[tag].append(fake_obj)

            for tag, obj_idx in sorted(to_remove, key=lambda x: x[1], reverse=True):
                struct[tag].pop(obj_idx)
            
            final_rxns.append(render_rxn_string(struct))

        new_entry = {
            "messages": [
                entry["messages"][0],
                {"content": "\n".join(final_rxns), "role": "assistant"}
            ],
            "images": entry["images"],
            "disturb_rxns": disturbed_rxn_ids # Record which equations were perturbed
        }
        new_data.append(new_entry)
            
    return new_data



if __name__ == '__main__':
    # Generate Prefix-Perturbed Reaction Parsing SFT data from 4240 real samples and 63600 aug. samples
    files = ["./sft_data_process/raw_data/train_downsampled_llm.json", "./sft_data_process/aug_data/train_downsampled_llm.json"]

    res = []
    for file_name in files:
        if os.path.exists(file_name):
            processed_data = process_json(file_name)
            res.extend(processed_data)

    output_name = "./sft_data_process/pprp_data/train_downsampled_llm_pprp.json"
    os.makedirs(os.path.dirname(output_name), exist_ok=True)

    with open(output_name, 'w', encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False, indent=4)





