import glob
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re
import argparse
import sys
import numpy as np
import random
from transformers import set_seed

sys.path.append("eval/")
from evaluater import ReactionEvaluator
import datetime

rp_instruct = """
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


def get_args(notebook=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./eval/SFT_Model/pretrainllm_sftall_lr1e-5_bs4_cosine_decouple_disturb_15d_checkpoint-47700')
    parser.add_argument('--eval_file_path', type=str, default='./dpo_data_process/dpo_data/train_downsampled_rxn_datasource_rel_aug.json')
    parser.add_argument('--choose_file_path', type=str, default='./dpo_data_process/dpo_data/train_downsampled_llm_datasource_rel_aug.json')
    parser.add_argument('--dpo_output_path', type=str, default='./dpo_data_process/dpo_data/train_downsampled_llm_dpo.json')
    parser.add_argument('--max_pixels', type=int, default=1000000)
    parser.add_argument('--max_new_tokens', type=int, default=4500)
    parser.add_argument('--infer_batch_size', type=int, default=1)
    parser.add_argument('--local-rank', type=int, default=-1)

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args


def clean_unwanted_tokens(text):
    # List of tokens to remove (built-in dialogue markers from the Qwen model)
    unwanted_tokens = [
        '<|im_end|>',
        '<|im_start|>',
        '<|endoftext|>',
        '<|end|>',
        '<|start|>',
    ]
    
    for token in unwanted_tokens:
        text = text.replace(token, '')

    text = text.strip()
    
    return text


def prepare_inputs(img_path_list, instruction, processor):
    if isinstance(instruction, str):
        instruction_list = [instruction] * len(img_path_list)
    else:
        instruction_list = instruction
        assert len(instruction_list) == len(img_path_list), "Instruction count must match image count"
    messages = []
    for img_path, instr in zip(img_path_list, instruction_list):
        messages.append([
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": instr}
                ]
            }
        ])

    texts = [
        processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        for msg in messages
    ]

    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=texts,
        images=image_inputs,
        padding=True,
        padding_side='left',
        return_tensors="pt",
    )

    device_index = dist.get_rank() % torch.cuda.device_count()
    return inputs.to(f"cuda:{device_index}")


def convert_pred_special_token(predictions, eval_data):
    type_to_id = {'<mol>': 1, "<txt>": 2}
    type_to_ident = {'<mol>': "[Mol]", "<txt>": "[Txt]"}
    
    res = []
    for pred, raw_data in zip(predictions, eval_data):
        image_width, image_height = raw_data["width"], raw_data["height"]
        converted_pred = []
        
        # If pred is a string, first parse the special-token format
        if isinstance(pred, str):
            # Split by <rxn> to get each reaction
            rxn_list = pred.split('<rxn>')[1:]  # Skip the first string (it may be empty or AR)
            
            for rxn_str in rxn_list:
                converted_rxn = {
                    'reactants': [],
                    'conditions': [],
                    'products': []
                }
                rxn_str = rxn_str.strip()
                
                # Parse the reactant section
                if '<rct>' in rxn_str:
                    rct_start = rxn_str.index('<rct>') + 5
                    rct_end = rxn_str.index('<cnd>') if '<cnd>' in rxn_str else len(rxn_str)
                    rct_str = rxn_str[rct_start:rct_end]
                    
                    # Use regex to extract coordinates and types
                    pattern = r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(<mol>|<txt>)'
                    matches = re.findall(pattern, rct_str)
                    try:
                        for match in matches:
                            x1, y1, x2, y2, mol_type = match
                            converted_rxn['reactants'].append({
                                'category': type_to_ident[mol_type],
                                'bbox': (float(x1) / image_width, float(y1) / image_height, 
                                        float(x2) / image_width, float(y2) / image_height),
                                'category_id': type_to_id[mol_type]
                            })   
                    except Exception as e:
                        print(f"parse reactant err: {e}")
                        print(f"parse reactant err: rct_str: {rct_str}, rxn_str: {rxn_str}")
                    if len(matches) == 0:
                        print(f"parse reactant len(matches) == 0: rct_str: {rct_str}, rxn_str: {rxn_str}")
                
                # Parse the condition section
                if '<cnd>' in rxn_str:
                    cnd_start = rxn_str.index('<cnd>') + 5
                    cnd_end = rxn_str.index('<prd>') if '<prd>' in rxn_str else len(rxn_str)
                    cnd_str = rxn_str[cnd_start:cnd_end]
                    
                    pattern = r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(<mol>|<txt>)'
                    matches = re.findall(pattern, cnd_str)
                    try:
                        for match in matches:
                            x1, y1, x2, y2, mol_type = match
                            converted_rxn['conditions'].append({
                                'category': type_to_ident[mol_type],
                                'bbox': (float(x1) / image_width, float(y1) / image_height, 
                                        float(x2) / image_width, float(y2) / image_height),
                                'category_id': type_to_id[mol_type]
                            })         
                    except Exception as e:
                        print(f"parse condition err: {e}")
                        print(f"parse condition err: cnd_str: {cnd_str}, rxn_str: {rxn_str}")
                        
                    if len(matches) == 0 and len(cnd_str) > 0:
                        print(f"parse condition len(matches) == 0: cnd_str: {cnd_str}, rxn_str: {rxn_str}")
                
                # Parse the product section
                if '<prd>' in rxn_str:
                    prd_start = rxn_str.index('<prd>') + 5
                    prd_str = rxn_str[prd_start:]
                    
                    pattern = r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(<mol>|<txt>)'
                    matches = re.findall(pattern, prd_str)
                    try:
                        for match in matches:
                            x1, y1, x2, y2, mol_type = match
                            converted_rxn['products'].append({
                                'category': type_to_ident[mol_type],
                                'bbox': (float(x1) / image_width, float(y1) / image_height, 
                                        float(x2) / image_width, float(y2) / image_height),
                                'category_id': type_to_id[mol_type]
                            })
                    except Exception as e:
                        print(f"parse product err: {e}")
                        print(f"parse product err: prd_str: {prd_str}, rxn_str: {rxn_str}")
                    
                    if len(matches) == 0:
                        print(f"parse product len(matches) == 0: prd_str: {prd_str}, rxn_str: {rxn_str}")
                
                converted_pred.append(converted_rxn)
        else:
            print("pred is not a string")
        
        res.append(converted_pred)
    return res


def load_json_files(folder_path):
    # Build search pattern, match all .json files
    file_pattern = os.path.join(folder_path, "*.json")
    file_list = glob.glob(file_pattern)
    
    all_data = []
    
    print(f"Start reading folder: {folder_path}, found {len(file_list)} files.")
    
    # Use tqdm to display progress bar (can remove tqdm and loop directly if not installed)
    for file_path in tqdm(file_list, desc="Reading..."):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_data.append(data)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            
    return all_data


def seed_everything(seed=42):
    set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args=None):
    if args is None:
        args = get_args()
    seed_everything(seed=42)

    timeout = datetime.timedelta(minutes=240)
    dist.init_process_group(backend='nccl', timeout=timeout)
    device_index = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device_index)
    
    # 1. Load the model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={'': device_index}
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    # 2. Load the data
    with open(args.eval_file_path, 'r') as f:
        eval_json = json.load(f)
    with open(args.choose_file_path, 'r') as f:
        choose_data = json.load(f)
    
    image_paths = [img["file_name"] for img in eval_json["images"]]    
    sampler = DistributedSampler(image_paths, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    dataloader = DataLoader(image_paths, batch_size=args.infer_batch_size, sampler=sampler)

    # 3. Inference
    local_results = []
    sampler_indices = list(sampler)
    
    for i, batch_paths in enumerate(tqdm(dataloader, desc=f"GPU {dist.get_rank()} Processing", disable=not dist.get_rank() == 0)):
        inputs = prepare_inputs(batch_paths, rp_instruct, processor)
        with torch.inference_mode():
            generate_model = getattr(model, "module", model)
            generated_ids = generate_model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=True,
                temperature=0.8,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )

            prompt_len = inputs.input_ids.shape[-1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            batch_responses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )

            batch_responses = [clean_unwanted_tokens(response) for response in batch_responses]
            
            for j, res in enumerate(batch_responses):
                global_idx = sampler_indices[i * args.infer_batch_size + j]
                local_results.append({"idx": global_idx, "img_path": batch_paths[j], "res": res})

                cache_dir = "./dpo_data_process/cache_infer_results"
                os.makedirs(cache_dir, exist_ok=True)
                with open(os.path.join(cache_dir, str(global_idx)+".json"), 'w', encoding='utf-8') as f:
                    json.dump({"idx": global_idx, "img_path": batch_paths[j], "res": res}, f, ensure_ascii=False, indent=4)


    # 4. Gather results
    all_gathered = [None] * dist.get_world_size()
    dist.all_gather_object(all_gathered, local_results)

    if dist.get_rank() == 0:
        # flat_results = load_json_files(cache_dir) # if all_gather_object error, un-comment this line
        flat_results = [item for sublist in all_gathered for item in sublist]
        unique_map = {item['idx']: (item['img_path'], item['res']) for item in flat_results}
        flat_results = [{"idx": i, "res": unique_map[i][1]} for i in sorted(unique_map.keys())]

        for i in sorted(unique_map.keys()):
            assert unique_map[i][0] == eval_json["images"][i]["file_name"], print("####idx mismatch error#####")
                        
        # with open(args.dpo_output_path.replace('.json', '_raw.json') , 'w', encoding='utf-8') as f:
        #     json.dump(flat_results, f, ensure_ascii=False, indent=4)
        
        # 5. Format conversion and evaluation-based filtering
        evaluator = ReactionEvaluator()
        dpo_data_list = []
        
        # 6. Convert all prediction results in advance
        all_rejected_texts = [x['res'] for x in flat_results]
        all_pred_converted = convert_pred_special_token(all_rejected_texts, eval_json["images"])
        
        print("Evaluating and filtering DPO samples...")
        for idx in tqdm(range(len(eval_json["images"]))):
            gold_img = eval_json["images"][idx]
            pred_img = all_pred_converted[idx]
            
            # Metric computation
            gh, ph = evaluator.evaluate_image(gold_img, pred_img)
            gh_m, ph_m = evaluator.evaluate_image(gold_img, pred_img, mol_only=True, merge_condition=True)
            
            m1 = evaluator.compute_metrics(sum(gh), len(gh), sum(ph), len(ph))
            m2 = evaluator.compute_metrics(sum(gh_m), len(gh_m), sum(ph_m), len(ph_m))
            
            # Filtering criterion: Overall F1 below 0.8
            if m1['f1'] < 0.8:
                chosen_val = choose_data[idx]["messages"][1]["content"] 
                
                dpo_entry = {
                    "messages": [
                        {
                            "from": "user",
                            "value": choose_data[idx]["messages"][0]["content"] 
                        }
                    ],
                    "chosen": {
                        "from": "assistant",
                        "value": chosen_val
                    },
                    "rejected": {
                        "from": "assistant",
                        "value": all_rejected_texts[idx]
                    },
                    "images": choose_data[idx]["images"],
                    "overall": m1,
                    "mol_only": m2
                }
                dpo_data_list.append(dpo_entry)

        # 7. Save results
        with open(args.dpo_output_path, 'w', encoding='utf-8') as f:
            json.dump(dpo_data_list, f, ensure_ascii=False, indent=4)
        
        print(f"DPO Data Generation Complete. Saved {len(dpo_data_list)} samples to {args.dpo_output_path}")

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
