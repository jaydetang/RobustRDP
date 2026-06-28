import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
import json
from PIL import Image
from tqdm import tqdm
import re
import math
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import argparse
import sys
from evaluater import ReactionEvaluator
import datetime
# import json_repair


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
    parser.add_argument('--eval_file_path', type=str, default='/data1/jiantingtang/workspace/Visual-RFT/lisa_evaluation/stage_2/rxnscribe_data/dev.json')
    parser.add_argument('--model_path', type=str, default='/data1/jiantingtang/workspace/LLaMA-Factory/saves/Qwen2-VL-2B/sft_stage_2/st1steps400_epoch50_lr1e5')
    parser.add_argument('--max_pixels', type=int, default=1000000)
    parser.add_argument('--max_new_tokens', type=int, default=2048)
    parser.add_argument('--infer_batch_size', type=int, default=4)
    parser.add_argument('--pred_output_path', type=str, default='/data1/jiantingtang/workspace/Visual-RFT/lisa_evaluation/evaluate/predictions.json')
    parser.add_argument('--score_output_path', type=str, default='/data1/jiantingtang/workspace/Visual-RFT/lisa_evaluation/evaluate/scores.json')
    parser.add_argument('--per_sample_output_path', type=str, default=None, help='每个样本指标输出路径，如果为None则自动生成')
    parser.add_argument('--use_lora', action='store_true', help='是否使用LoRA adapter进行推理')
    parser.add_argument('--adapter_path', type=str, default=None, help='LoRA adapter路径，如果为None则使用model_path')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--task_type', type=str, default='rp', choices=['rp'], help='选择评测任务类型：rp')

    args = parser.parse_args([]) if notebook else parser.parse_args()
    return args

def clean_unwanted_tokens(text):
    """
    清理模型自带的特殊token，但保留训练时添加的自定义token
    """
    # 需要清理的token列表（Qwen模型自带的对话标记）
    unwanted_tokens = [
        '<|im_end|>',
        '<|im_start|>',
        '<|endoftext|>',
        '<|end|>',
        '<|start|>',
    ]
    
    for token in unwanted_tokens:
        text = text.replace(token, '')
    
    # 清理多余的空白字符
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

    return inputs.to(f"cuda:{dist.get_rank()}")

def convert_pred_special_token(predictions, eval_data):
    type_to_id = {'<mol>': 1, "<txt>": 2}
    type_to_ident = {'<mol>': "[Mol]", "<txt>": "[Txt]"}
    
    res = []
    for pred, raw_data in zip(predictions, eval_data):
        image_width, image_height = raw_data["width"], raw_data["height"]
        converted_pred = []
        
        # 如果pred是字符串，先解析特殊token格式
        if isinstance(pred, str):
            # 按<rxn>分割获取每个反应
            rxn_list = pred.split('<rxn>')[1:]  # 跳过第一个字符串(可能是空，也可能是AR)
            
            for rxn_str in rxn_list:
                converted_rxn = {
                    'reactants': [],
                    'conditions': [],
                    'products': []
                }
                rxn_str = rxn_str.strip()
                
                # 解析反应物部分
                if '<rct>' in rxn_str:
                    rct_start = rxn_str.index('<rct>') + 5
                    rct_end = rxn_str.index('<cnd>') if '<cnd>' in rxn_str else len(rxn_str)
                    rct_str = rxn_str[rct_start:rct_end]
                    
                    # 使用正则表达式提取坐标和类型
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
                        print(f"parse reactant err: rct_str: {rct_str}, rxn_str: {rxn_str}")
                
                # 解析条件部分
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
                        print(f"parse condition err: cnd_str: {cnd_str}, rxn_str: {rxn_str}")
                
                # 解析产物部分
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
                        print(f"parse product err: prd_str: {prd_str}, rxn_str: {rxn_str}")
                
                converted_pred.append(converted_rxn)
        else:
            print("pred is not a string")
        
        res.append(converted_pred)
    return res


def main():
    args = get_args()

    task_instruction_map = {
        'rp': rp_instruct,
    }
    task_instruction = task_instruction_map[args.task_type]

    # Initialize the distributed environment
    timeout = datetime.timedelta(minutes=30)
    dist.init_process_group(backend='nccl', timeout=timeout)
    torch.cuda.set_device(args.local_rank)

    # 根据use_lora参数选择加载方式
    if args.use_lora:
        # LoRA模式：加载base模型 + adapter
        if dist.get_rank() == 0:
            print("="*50)
            print("使用 LoRA 模式进行推理")
            print(f"Base模型路径: {args.model_path}")
            adapter_path = args.adapter_path if args.adapter_path else args.model_path
            print(f"Adapter路径: {adapter_path}")
            print("="*50)
        
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={'': args.local_rank}  # 直接加载到指定GPU
        )
        
        # 加载adapter（不merge）
        adapter_path = args.adapter_path if args.adapter_path else args.model_path
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False
        )
        
        # 推理时不需要DDP（模型参数无梯度）
        # model = DDP(model, device_ids=[args.local_rank])
        
        # processor从base模型路径加载
        processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)
    else:
        # 全参数模式：直接加载完整模型
        if dist.get_rank() == 0:
            print("="*50)
            print("使用全参数模式进行推理")
            print(f"模型路径: {args.model_path}")
            print("="*50)
        
        # 直接加载到指定GPU以节省内存
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map={'': args.local_rank}  # 直接加载到指定GPU
        )
        # 推理时不需要DDP（模型参数无梯度）
        # model = DDP(model, device_ids=[args.local_rank])
        
        processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=args.max_pixels)

    raw_eval_data = json.load(open(args.eval_file_path, 'r'))
    image_path_list = [item["file_name"] for item in raw_eval_data["images"]]

    
    if dist.get_rank() == 0:
        print(f"Total samples to evaluate: {len(image_path_list)}")

    # Create a DistributedSampler
    sampler = DistributedSampler(image_path_list, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False)
    dataloader = DataLoader(image_path_list, batch_size=args.infer_batch_size, sampler=sampler)

    # ==== 推理阶段：每个GPU保存自己的结果到临时文件 ====
    responses = []
    response_indices = []  # 记录每个response对应的全局索引
    
    model.eval()
    local_idx = 0
    for batch_paths in tqdm(dataloader, desc=f"GPU {dist.get_rank()} Processing", disable=not dist.get_rank() == 0):
        inputs = prepare_inputs(batch_paths, task_instruction, processor)
        with torch.inference_mode():
            generate_model = getattr(model, "module", model)
            generated_ids = generate_model.generate(
                **inputs, 
                max_new_tokens=args.max_new_tokens, 
                do_sample=False,
                temperature=0.0,
                use_cache=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
            # 左填充批处理时，生成序列会携带完整 prompt（含 pad）；需要统一截掉整段输入长度
            prompt_len = inputs.input_ids.shape[-1]
            generated_ids_trimmed = generated_ids[:, prompt_len:]
            batch_responses = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            # 清理不需要的特殊token，保留自定义token
            batch_responses = [clean_unwanted_tokens(response) for response in batch_responses]
            sampler_list = list(sampler)  # 构造一次，保持稳定
            # 记录全局索引
            for i, response in enumerate(batch_responses):
                global_idx = sampler_list[local_idx]
                responses.append(response)
                response_indices.append(global_idx)
                local_idx += 1
            
        del inputs, generated_ids, generated_ids_trimmed
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    # 每个GPU将结果保存到临时文件
    temp_dir = os.path.join(os.path.dirname(args.pred_output_path), 'temp_predictions')
    os.makedirs(temp_dir, exist_ok=True)
    temp_file = os.path.join(temp_dir, f'rank_{dist.get_rank()}_predictions.json')
    
    with open(temp_file, 'w') as f:
        json.dump({'indices': response_indices, 'responses': responses}, f)
    
    if dist.get_rank() == 0:
        print(f"GPU {dist.get_rank()} saved {len(responses)} predictions to {temp_file}")
    
    # 释放推理相关的内存
    del model, processor, responses, response_indices
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    # 等待所有进程完成
    dist.barrier()

    # ==== 评估阶段：只在rank 0进行，分批处理 ====
    if dist.get_rank() == 0:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        print("\n" + "="*50)
        print("开始合并和评估结果...")
        print("="*50)
        
        # 创建输出目录
        os.makedirs(os.path.dirname(args.pred_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(args.score_output_path), exist_ok=True)
        
        # 读取所有GPU的结果并重新排序
        print("正在读取所有GPU的结果...")
        reordered_responses = [None] * len(image_path_list)
        
        for rank in range(dist.get_world_size()):
            temp_file = os.path.join(temp_dir, f'rank_{rank}_predictions.json')
            with open(temp_file, 'r') as f:
                data = json.load(f)
                for idx, response in zip(data['indices'], data['responses']):
                    reordered_responses[idx] = response
        
        print(f"成功读取 {len([r for r in reordered_responses if r is not None])} 个预测结果")
        
        # 分批转换和评估
        batch_size = 1000  # 每批处理1000个样本
        num_batches = math.ceil(len(reordered_responses) / batch_size)
        
        print(f"\n开始分批处理，共 {num_batches} 批，每批 {batch_size} 个样本")
        
        all_predictions = []
        evaluator = ReactionEvaluator()
        
        # 用于累积评估统计
        total_stats = {
            'gold_hits': 0, 
            'gold_total': 0, 
            'pred_hits': 0, 
            'pred_total': 0
        }
        total_stats_mol_only = {
            'gold_hits': 0, 
            'gold_total': 0, 
            'pred_hits': 0, 
            'pred_total': 0
        }
        
        # 存储每个样本的指标
        per_sample_metrics = []
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(reordered_responses))
            
            print(f"\n处理批次 {batch_idx + 1}/{num_batches} (样本 {start_idx} 到 {end_idx})...")
            
            # 获取当前批次的数据
            batch_responses = reordered_responses[start_idx:end_idx]
            batch_images = raw_eval_data['images'][start_idx:end_idx]
            
            # 转换预测格式
            batch_predictions = convert_pred_special_token(batch_responses, batch_images)
            all_predictions.extend(batch_predictions)
            
            # 评估当前批次并累积统计
            for gold_image, pred_image in zip(batch_images, batch_predictions):
                # 整体评估
                gh, ph = evaluator.evaluate_image(gold_image, pred_image)
                total_stats['gold_hits'] += sum(gh)
                total_stats['gold_total'] += len(gh)
                total_stats['pred_hits'] += sum(ph)
                total_stats['pred_total'] += len(ph)
                
                # mol_only 评估
                gh_mol, ph_mol = evaluator.evaluate_image(gold_image, pred_image, 
                                                          mol_only=True, merge_condition=True)
                total_stats_mol_only['gold_hits'] += sum(gh_mol)
                total_stats_mol_only['gold_total'] += len(gh_mol)
                total_stats_mol_only['pred_hits'] += sum(ph_mol)
                total_stats_mol_only['pred_total'] += len(ph_mol)
                
                # 计算每个样本的指标
                sample_metrics = {
                    'file_name': gold_image.get('file_name', 'unknown'),
                    'overall': evaluator.compute_metrics(sum(gh), len(gh), sum(ph), len(ph)),
                    'mol_only': evaluator.compute_metrics(sum(gh_mol), len(gh_mol), sum(ph_mol), len(ph_mol)),
                }
                per_sample_metrics.append(sample_metrics)
            
            # 释放当前批次的内存
            del batch_responses, batch_predictions
            gc.collect()
            
            print(f"批次 {batch_idx + 1} 完成")
        
        # 计算最终指标
        print("\n" + "="*50)
        print("计算最终评估指标...")
        print("="*50)
        
        overall_metrics = evaluator.compute_metrics(
            total_stats['gold_hits'], 
            total_stats['gold_total'], 
            total_stats['pred_hits'], 
            total_stats['pred_total']
        )
        
        mol_only_metrics = evaluator.compute_metrics(
            total_stats_mol_only['gold_hits'], 
            total_stats_mol_only['gold_total'], 
            total_stats_mol_only['pred_hits'], 
            total_stats_mol_only['pred_total']
        )
        
        precision, recall, f1 = overall_metrics['precision'], overall_metrics['recall'], overall_metrics['f1']
        print(f'\n整体结果:')
        print(f'  Precision: {precision:.4f}')
        print(f'  Recall: {recall:.4f}')
        print(f'  F1: {f1:.4f}')
        
        # 保存结果
        print(f"\n保存预测结果到 {args.pred_output_path}...")
        with open(args.pred_output_path, 'w') as f:
            json.dump(all_predictions, f, indent=4)
        
        results = {
            'overall': overall_metrics,
            'overall_stats': total_stats,
            'mol_only': mol_only_metrics,
            'mol_only_stats': total_stats_mol_only
        }
        
        print(f"保存评估分数到 {args.score_output_path}...")
        with open(args.score_output_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # 保存每个样本的指标
        print(f"保存每个样本指标到 {args.per_sample_output_path}...")
        with open(args.per_sample_output_path, 'w') as f:
            json.dump(per_sample_metrics, f, indent=4)
        
        # 清理临时文件
        print("\n清理临时文件...")
        import shutil
        shutil.rmtree(temp_dir)
        
        print("\n" + "="*50)
        print("评估完成！")
        print("="*50)

    dist.destroy_process_group()

if __name__ == '__main__':
    main()
