import os
import json
import threading

from utils.down_sample_rxn import down_sample_main


def process_bbox_id(data):
    for item in data["images"]:
        bboxes = item.get("bboxes", [])
        reactions = item.get("reactions", [])
        file_name = item.get("file_name", "")

        # 1. Check if IDs are already consecutive integers from 0 to n-1
        current_ids = [box["id"] for box in bboxes]
        expected_ids = list(range(len(bboxes)))

        if current_ids == expected_ids:
            # Skip if the requirement is met
            continue

        # 2. If not satisfied, print the filename and start modifying
        print(f"Updating IDs for file: {file_name}")

        # Create mapping from old ID to new ID
        id_map = {}
        for index, box in enumerate(bboxes):
            old_id = box["id"]
            new_id = index
            id_map[old_id] = new_id
            box["id"] = new_id  # Update the id inside the bbox

        # 3. Update references in reactions
        for reaction in reactions:
            # Update reactants
            if "reactants" in reaction:
                reaction["reactants"] = [id_map[rid] for rid in reaction["reactants"]]
            
            # Update conditions
            if "conditions" in reaction:
                reaction["conditions"] = [id_map[cid] for cid in reaction["conditions"]]
            
            # Update products
            if "products" in reaction:
                reaction["products"] = [id_map[pid] for pid in reaction["products"]]

    return data


def gen_processed_val_data(input_folder, output_filename):
    # Define standard category mapping
    categories = [
        {"id": 1, "name": "structure"},
        {"id": 2, "name": "text"},
        {"id": 3, "name": "identifier"},
        {"id": 4, "name": "supplement"}
    ]
    
    merged_data = {
        "categories": categories,
        "images": []
    }
    
    # Traverse all files in the folder
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)                    
                merged_data["images"].append(content)

    merged_data = process_bbox_id(merged_data)

    # Save the result as a combined JSON file
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, indent=4, ensure_ascii=False)
    
    print(f"Processing completed! A total of {len(merged_data['images'])} files merged, output: {output_filename}")


data_lock = threading.Lock()

augment_data = {
    "categories": [
        {"id": 1, "name": "structure"},
        {"id": 2, "name": "text"},
        {"id": 3, "name": "identifier"},
        {"id": 4, "name": "supplement"},
    ],
    "roles": [
        {"id": 1, "name": "reactants"},
        {"id": 2, "name": "conditions"},
        {"id": 3, "name": "products"},
    ],
    "images": [],
}


if __name__ == "__main__":
    # 将 ./raw_val_data/RxnScribe_test中的原始数据进行转换，对图片进行缩放，转换为RobustRDP的格式
    down_sample_main("./raw_val_data/RxnScribe_test/images_eval", './raw_val_data/RxnScribe_test/dev.json', mode="dev")