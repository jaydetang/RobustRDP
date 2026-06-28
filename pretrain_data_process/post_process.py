import json
import os
import math
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

# 目标尺寸
target_size = 1000 # 假设 target_size 为 896 (常见 28 的倍数)，可根据需求调整

reaction_parsing_instruct = "<image> \nEach image contains one or more chemical reaction flowcharts, some of which may be horizontally flipped or rotated by 90°.\nYour task is to identify and extract the structured information of each reaction, including reactants, products, and reaction conditions.\n\nOutput Format:\nThe chemical reaction flowchart contains multiple reaction equations. Please output the bounding box coordinates and content of all reactants, conditions, and products in each equation using the following special token format:\n<rxn><rct>x1 y1 x2 y2<mol>x1 y1 x2 y2<txt><cnd>x1 y1 x2 y2<txt><prd>x1 y1 x2 y2<mol>\n\nFormat specifications:\n- Each reaction starts with <rxn> token\n- Reactants section starts with <rct> token\n- Conditions section starts with <cnd> token  \n- Products section starts with <prd> token\n- Each chemical entity is represented by: x1 y1 x2 y2 <type>\n  - x1, y1: top-left corner coordinates (floating point numbers)\n  - x2, y2: bottom-right corner coordinates (floating point numbers)\n  - <type>: either <mol> for molecular structures or <txt> for text blocks\n- Multiple reactions should be output sequentially using the same format\n- The reactions should be organized in the order they appear in the image, from left to right and top to bottom\n- If a section (reactants/conditions/products) has no entities, the section token should still be included\n\nExample for multiple reactions:\n<rxn><rct>10 20 50 60 <mol><cnd>70 25 120 45 <txt><prd>140 22 180 62 <mol>\n<rxn><rct>200 30 240 70 <mol><cnd><prd>280 28 320 68 <mol>\n"

def downsample_image(image_path, output_dir, target_size):
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    original_width, original_height = img.size
    max_possible_width = target_size // 28 * 28
    max_possible_height = target_size // 28 * 28

    scale_x = min(max_possible_width / original_width, 1.0)
    scale_y = min(max_possible_height / original_height, 1.0)
    
    scale = math.sqrt(scale_x * scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    new_width = (new_width // 28) * 28
    new_height = (new_height // 28) * 28
        
    scale_x_final = new_width / original_width
    scale_y_final = new_height / original_height
    downsampled_image = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_name = os.path.basename(image_path)
    output_path = os.path.join(output_dir, file_name)

    downsampled_image.save(output_path, quality=95)
    
    return (scale_x_final, scale_y_final), (new_width, new_height), output_path


def process_single_entry(entry, output_dir, target_size, instruct):
    """处理单个 entry 的逻辑，提取自原循环体"""
    # 1. 替换指令
    entry["messages"][0]["content"] = instruct
    
    image_path = entry["images"][0]
    # 4. 图片下采样并保存
    scales, new_dims, output_path = downsample_image(image_path, output_dir, target_size)
    scale_x, scale_y = scales
    new_width, new_height = new_dims
    entry["images"][0] = output_path
    
    with Image.open(image_path) as img:
        orig_w, orig_h = img.size

    content_str = entry["messages"][1]["content"]
    json_str = content_str.replace("<answer>", "").replace("</answer>", "").strip()
    reactions = json.loads(json_str)
    
    final_output_parts = []

    for rxn in reactions:
        # 3. 移动 conditions 中的 molecule 到 reactants
        new_cond = []
        for c in rxn["conditions"]:
            if c["content"] == "molecule":
                rxn["reactants"].append(c)
            else:
                new_cond.append(c)
        rxn["conditions"] = new_cond

        def format_entities(entities):
            res = ""
            for e in entities:
                x1 = (e["bbox"][0] / 1000.0) * orig_w * scale_x
                y1 = (e["bbox"][1] / 1000.0) * orig_h * scale_y
                x2 = (e["bbox"][2] / 1000.0) * orig_w * scale_x
                y2 = (e["bbox"][3] / 1000.0) * orig_h * scale_y

                # --- 核心修复：限制范围在 [0, new_width/height] ---
                x1 = max(0, min(new_width, int(round(x1))))
                y1 = max(0, min(new_height, int(round(y1))))
                x2 = max(0, min(new_width, int(round(x2))))
                y2 = max(0, min(new_height, int(round(y2))))
                
                type_tag = "<mol>" if e["content"] == "molecule" else "<txt>"
                res += f"{int(round(x1))} {int(round(y1))} {int(round(x2))} {int(round(y2))}{type_tag}"
            return res

        rxn_str = "<rxn>"
        rxn_str += "<rct>" + format_entities(rxn["reactants"])
        rxn_str += "<cnd>" + format_entities(rxn["conditions"])
        rxn_str += "<prd>" + format_entities(rxn["products"])
        final_output_parts.append(rxn_str)

    entry["messages"][1]["content"] = "\n".join(final_output_parts)
    return entry


def process_data_parallel(data_list, num_workers=8):
    processed_list = []
    
    # 使用 ProcessPoolExecutor 并通过 submit 提交任务
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # 提交所有任务
        futures = [
            executor.submit(process_single_entry, entry, output_dir, target_size, reaction_parsing_instruct) 
            for entry in data_list
        ]
        
        # 使用 tqdm 结合 as_completed 监控进度
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
            try:
                result = future.result()
                processed_list.append(result)
            except Exception as e:
                print(f"Error processing entry: {e}")
                
    return processed_list


if __name__ == "__main__":
    data_types = ["single_line", "multi_line", "branch", "cycle"]

    for data_type in data_types:
        print(f"Processing {data_type}...")

        # 加载数据
        input_path = f"./pretrain_data_process/images/{data_type}.json"
        with open(input_path, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        output_dir = f"./pretrain_data_process/images_pretrain_resized/{data_type}_resized"
        os.makedirs(output_dir, exist_ok=True)

        # 并行处理
        final_data = process_data_parallel(original_data, num_workers=64)

        # 写入结果
        output_json_path = f"./pretrain_data_process/images_pretrain_resized/{data_type}_resized.json"
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)

        print(f"Finished {data_type}, saved to {output_json_path}")

    # 合并所有类型的 JSON 文件
    print("\n" + "=" * 30)
    print("Merging all types into one file...")

    input_files = [
        "./pretrain_data_process/images_pretrain_resized/single_line_resized.json",
        "./pretrain_data_process/images_pretrain_resized/multi_line_resized.json",
        "./pretrain_data_process/images_pretrain_resized/branch_resized.json",
        "./pretrain_data_process/images_pretrain_resized/cycle_resized.json"
    ]

    output_file = "./pretrain_data_process/pretrain_downsampled_llm_6w.json"

    combined_data = []

    for file_path in input_files:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        combined_data.extend(data)
                        print(f"Successfully loaded {file_path}, count: {len(data)}")
                    else:
                        print(f"Warning: {file_path} is not a list, skipping.")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")

    # 保存合并后的结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(combined_data, f, indent=4, ensure_ascii=False)

    print("-" * 30)
    print(f"Total entries in combined file: {len(combined_data)}")
    print(f"Saved to: {output_file}")