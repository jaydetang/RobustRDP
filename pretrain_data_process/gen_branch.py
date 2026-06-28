import os
import sys
import pandas as pd
import random
import json
from PIL import Image, ImageDraw, ImageFont
import cv2
import math
from indigo import Indigo
from utils import generate_indigo_image, draw_cond_img, get_fonts
from concurrent.futures import ProcessPoolExecutor, as_completed

instruct = """The chemical reaction flowchart contains multiple reaction equations. Please output the bounding box coordinates and content of all reactants, conditions, and products in each equation in the following JSON format:
[
    {
        "reactants": [{"bbox": [x1, y1, x2, y2], "content": "molecule"}],
        "conditions": [{"bbox": [x1, y1, x2, y2], "content": "text"}],
        "products": [{"bbox": [x1, y1, x2, y2], "content": "molecule"}]
    }
]
The bounding box coordinates are represented as [x1, y1, x2, y2], where (x1, y1) is the top-left corner, and (x2, y2) is the bottom-right corner of the bounding box.
- If an object represents a molecule, the "content" field should be filled with the placeholder string "molecule".  
- If an object represents a text block, the "content" field should be filled with the placeholder string "text".
The equations in the JSON should be organized in the order they appear in the image, from left to right and top to bottom.
Directly output the JSON and the JSON should be enclosed within <answer> </answer> tag, i.e., <answer> JSON </answer>."""


MOL_SIZE_RANGE = (100, 160)

# 初始化 Indigo
indigo = Indigo()

# 读取分子CSV文件
df = pd.read_csv('./pretrain_data_process/raw_data/pubchem/train_1m.csv')
smiles_list = df['SMILES'].dropna().tolist()

# 读取条件TXT文件
cond_list = []
with open('./pretrain_data_process/raw_data/conditions.txt', 'r', encoding='utf-8') as file:
    for line in file:
        line_content = eval(line.strip())
        joined_string = ' '.join(line_content).split(' ')[: 15]
        joined_string = ' '.join(joined_string)
        cond_list.append(joined_string)

# 字体库
font_dir = './pretrain_data_process/raw_data/ttf-ms-win10'
font_path_list = get_fonts(font_dir)


# 随机抽取N个分子
def get_random_smiles(n):
    return random.sample(smiles_list, n)

# 用Indigo生成分子图
def draw_molecule(smiles, mol_size):
    max_iterations = 20
    while max_iterations > 0:
        image, smiles, graph, success = generate_indigo_image(
            smiles, mol_augment=True, default_option=False,
            shuffle_nodes=False, pseudo_coords=False, include_condensed=True,
            image_width=mol_size)
        if success:
            break
        max_iterations -= 1
    
    if not success:
        print('draw mol fail')
        for i in range(10):
            image, smiles, graph, success = generate_indigo_image(
                "CC(C1CC2CCC1C2)N3C(=NN=C3SCC(=O)N4CCN(CC4)C5=CC=CC=C5)C6=CC=CS6", 
                mol_augment=True, default_option=False,
                shuffle_nodes=False, pseudo_coords=False, include_condensed=True,
                image_width=mol_size)
            if success:
                break
    
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    return image


# 绘制箭头
def draw_arrow(draw, start, end, color=(0,0,0)):
    draw.line([start, end], fill=color, width=random.randint(1, 3))
    arrow_length = 10
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    p1 = (end[0] - arrow_length * math.cos(angle - math.pi/6),
          end[1] - arrow_length * math.sin(angle - math.pi/6))
    p2 = (end[0] - arrow_length * math.cos(angle + math.pi/6),
          end[1] - arrow_length * math.sin(angle + math.pi/6))
    draw.polygon([end, p1, p2], fill=color)


# 随机生成反应条件文本
def random_condition_text():
    return random.sample(cond_list, 1)

# 生成单个反应方程
def generate_reaction():
    num_mols = random.choice([1,1,1,1,1,1,1,2,2,2])
    mols = get_random_smiles(num_mols)
    
    num_conds = random.randint(0, 2)
    conds = []
    for i in range(num_conds):
        if random.random() < 0.7:  # 70%概率条件是文本
            condition = random_condition_text()[0]
            condition_is_text = True
        else:
            condition = get_random_smiles(1)[0]
            condition_is_text = False
        conds.append((condition, condition_is_text))
        
    return mols, conds

# 布局函数（以单行为例，其他类似扩展）
def arrange_reactions(canvas, start_x, start_y, reactions):
    draw = ImageDraw.Draw(canvas)
    json_output = []
    new_reactions = []
    i = 0
    while i<len(reactions):
        line_rxn_num = random.randint(2, 3)
        new_reactions.append(reactions[i:i+line_rxn_num])
        i+=line_rxn_num
    if len(new_reactions[-1]) == 1:
        del new_reactions[-1]
    x = start_x
    y = start_y
    arrow_color = (0, 0, 0)
    grid_pos = [[[] for i in range(3)] for i in range(len(new_reactions))]
    grid_len = [len(item) for item in new_reactions]
    arrow_direc_list = []
    for line_idx, reactions in enumerate(new_reactions):
        arrow_direc = random.choice([True, False])  # True向右
        arrow_direc_list.append(arrow_direc)
        list_mols, list_conds = [], []
        for rxn_idx, (reactants, conditions) in enumerate(reactions):
            # Reactants
            list_mols_cord = []
            for i, smi in enumerate(reactants):
                mol_img = draw_molecule(smi, random.randint(*MOL_SIZE_RANGE))
                mol_w, mol_h = mol_img.size
                tmp_y = y+random.randint(5, 10)
                canvas.paste(mol_img, (x, tmp_y))
                bbox = [x, tmp_y, x+mol_w, tmp_y+mol_h]
                grid_pos[line_idx][rxn_idx].append(bbox)
                list_mols_cord.append({"bbox": bbox, "content": "molecule"})
                x += mol_w + random.randint(20, 30)
                # 加号
                if i != len(reactants) - 1:
                    draw.text((x, y + mol_h // 2 - 20), "+", fill='black', 
                            font=ImageFont.truetype('/data1/jiantingtang/workspace/Visual-RFT/lisa_evaluation/stage_0/raw_data/ttf-ms-win10/arial.ttf', 
                                                    random.randint(15, 25)))
                    x += random.randint(20, 30)
            list_mols.append(list_mols_cord)

            if rxn_idx == len(reactions)-1:
                continue
            
            if arrow_direc:
                arrow_start = (x, y + mol_h // 2)
                arrow_len = random.randint(120, 150)
                x += arrow_len
                arrow_end = (x, y + mol_h // 2)
                draw_arrow(draw, arrow_start, arrow_end, color=arrow_color)
            else:
                arrow_end = (x, y + mol_h // 2)
                arrow_len = random.randint(120, 150)
                x += arrow_len
                arrow_start = (x, y + mol_h // 2)
                draw_arrow(draw, arrow_start, arrow_end, color=arrow_color)

            # Condition
            list_conds_cord = []
            for cond_idx, (condition, condition_is_text) in enumerate(conditions):
                if condition_is_text:
                    text_img = draw_cond_img(condition, font_path_list)
                    text_w, text_h = text_img.size
                    text_x = x - arrow_len//2 - text_w//2
                    if len(conditions)==1:
                        if random.choice([True, False]):
                            text_y = arrow_start[1] - text_h - random.randint(5, 10)
                        else:
                            text_y = arrow_start[1] + random.randint(5, 10)
                    else:
                        if cond_idx == 0:
                            text_y = arrow_start[1] - text_h - random.randint(5, 10)
                        else:
                            text_y = arrow_start[1] + random.randint(5, 10)
                    canvas.paste(text_img, (text_x, text_y))
                    list_conds_cord.append({"bbox": [text_x, text_y, text_x+text_w, text_y+text_h], "content": "text"})
                else:
                    mol_img = draw_molecule(condition, random.randint(*MOL_SIZE_RANGE)//2)
                    mol_w, mol_h = mol_img.size
                    mol_x = x - arrow_len//2 - mol_w//2
                    if len(conditions)==1:
                        if random.choice([True, False]):
                            mol_y = arrow_start[1] - mol_h - random.randint(5, 10)
                        else:
                            mol_y = arrow_start[1] + random.randint(5, 10)
                    else:
                        if cond_idx == 0:
                            mol_y = arrow_start[1] - mol_h - random.randint(5, 10)
                        else:
                            mol_y = arrow_start[1] + random.randint(5, 10)
                    canvas.paste(mol_img, (mol_x, mol_y))
                    bbox = [mol_x, mol_y, mol_x+mol_w, mol_y+mol_h]
                    list_conds_cord.append({"bbox": bbox, "content": "molecule"})
            x += random.randint(20, 30)
            list_conds.append(list_conds_cord)

        if arrow_direc == False:
            list_mols = list_mols[::-1]
            list_conds = list_conds[::-1]
        for i in range(len(list_mols)-1):
            reaction_entry = {"reactants": list_mols[i], "conditions": list_conds[i], "products": list_mols[i+1]}
            json_output.append(reaction_entry)
        if line_idx != len(new_reactions)-1:
            json_output.append(None)
        
        x = start_x + random.randint(-30, 50)
        y = y + random.randint(150, 170)

    for i in range(len(arrow_direc_list)-1):
        reaction_entry = {"reactants": [], "conditions": [], "products": []}
        if (arrow_direc_list[i] and arrow_direc_list[i+1]) or (not arrow_direc_list[i] and arrow_direc_list[i+1]):
            if len(grid_pos[i][0]) == 1:
                x1, y1, x2, y2 = grid_pos[i][0][0]
                arrow_start = ((x1+x2)//2, y2+random.randint(2, 5))
                reaction_entry["reactants"] = [{"bbox": (x1, y1, x2, y2), "content": "molecule"}]
            else:
                x11, y11, x12, y12 = grid_pos[i][0][0]
                x21, y21, x22, y22 = grid_pos[i][0][1]
                arrow_start = ((x11+x22)//2, y12+random.randint(-5, 5))
                reaction_entry["reactants"] = [{"bbox": (x11, y11, x12, y12), "content": "molecule"},
                                               {"bbox": (x21, y21, x22, y22), "content": "molecule"}]

            if len(grid_pos[i+1][0]) == 1:
                x1, y1, x2, y2 = grid_pos[i+1][0][0]
            else:
                reactant_idx = random.randint(0, 1)
                x1, y1, x2, y2 = grid_pos[i+1][0][reactant_idx]
            arrow_end = ((x1+x2)//2, y1-random.randint(2, 5))
            reaction_entry["products"] = [{"bbox": (x1, y1, x2, y2), "content": "molecule"}]

            json_insert(json_output, reaction_entry)
            draw_arrow(draw, arrow_start, arrow_end, color=arrow_color)
        else:
            line_obj_idx = grid_len[i]-1
            if len(grid_pos[i][line_obj_idx]) == 1:
                x1, y1, x2, y2 = grid_pos[i][line_obj_idx][0]
                arrow_start = ((x1+x2)//2, y2+random.randint(2, 5))
                reaction_entry["reactants"] = [{"bbox": (x1, y1, x2, y2), "content": "molecule"}]
            else:
                x11, y11, x12, y12 = grid_pos[i][line_obj_idx][0]
                x21, y21, x22, y22 = grid_pos[i][line_obj_idx][1]
                arrow_start = ((x11+x22)//2, y12+random.randint(-5, 5))
                reaction_entry["reactants"] = [{"bbox": (x11, y11, x12, y12), "content": "molecule"},
                                               {"bbox": (x21, y21, x22, y22), "content": "molecule"}]

            if len(grid_pos[i+1][grid_len[i+1]-1]) == 1:
                x1, y1, x2, y2 = grid_pos[i+1][grid_len[i+1]-1][0]
            else:
                reactant_idx = random.randint(0, 1)
                x1, y1, x2, y2 = grid_pos[i+1][grid_len[i+1]-1][reactant_idx]
            arrow_end = ((x1+x2)//2, y1-random.randint(2, 5))
            reaction_entry["products"] = [{"bbox": (x1, y1, x2, y2), "content": "molecule"}]

            json_insert(json_output, reaction_entry)
            draw_arrow(draw, arrow_start, arrow_end, color=arrow_color)

    return json_output

# 添加branch分支
def json_insert(json_output, reaction_entry):
    cur = 0
    while json_output[cur] != None:
        cur += 1
    json_output[cur] = reaction_entry
    return json_output

# 裁剪画布
def crop(canvas):
    canvas_width, canvas_height = canvas.size

    new_x = canvas_width
    new_y = canvas_height

    for x in range(new_x-1, -1, -1):
        if any(canvas.getpixel((x, y)) != (255, 255, 255) for y in range(canvas_height)):
            new_x = x + 1
            break

    for y in range(new_y-1, -1, -1):
        if any(canvas.getpixel((x, y)) != (255, 255, 255) for x in range(canvas_width)):
            new_y = y + 1
            break

    canvas = canvas.crop((0, 0, 
                          min(canvas_width-1, new_x+random.randint(50, 100)), 
                          min(canvas_height-1, new_y+random.randint(50, 100))))
    return canvas


# 坐标归一化
def process_label(canvas, layout_json):
    canvas_width, canvas_height = canvas.size

    roles = ['reactants', 'conditions', 'products']
    for reaction in layout_json:
        for role in roles:
            new_items = []
            for item in reaction[role]:
                new_item = {'bbox': [min(int(item['bbox'][0] / canvas_width * 1000), 999),
                                    min(int(item['bbox'][1] / canvas_height * 1000), 999),
                                    min(int(item['bbox'][2] / canvas_width * 1000), 999),
                                    min(int(item['bbox'][3] / canvas_height * 1000), 999),
                                    ],
                            'content': item['content']}
                new_items.append(new_item)
            reaction[role] = new_items


# 主函数
def generate_image_and_json(output_dir, output_prefix):
    canvas_width = 2000
    canvas_height = 2000
    canvas = Image.new('RGB', (canvas_width, canvas_height), 'white')
    
    num_equations = random.randint(5, 8)
    reactions = [generate_reaction() for _ in range(num_equations)]
    layout_json = arrange_reactions(canvas, random.randint(50, 100), random.randint(50, 100), reactions)
    
    # 保存图像
    img_path = f"{output_dir}/{output_prefix}.png"
    canvas = crop(canvas)
    canvas.save(img_path)
    process_label(canvas, layout_json)
    
    res = {
        "messages": [
                {
                    "content": f"<image>\n {instruct}",
                    "role": "user"
                },
                {
                    "content": '<answer> ' + json.dumps(layout_json) + ' </answer>',
                    "role": "assistant"
                },
            ],
        "images": [
            img_path
        ]
    }

    print('finished!')
    return res



if __name__ == '__main__':
    IMAGE_NUM = 20
    OUTPUT_DIR = './pretrain_data_process/images/branch'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX = 'branch_{}'    

    all_json = [None]*IMAGE_NUM
    with ProcessPoolExecutor(max_workers=120) as executor:
        futures = [executor.submit(generate_image_and_json, OUTPUT_DIR, PREFIX.format(i)) for i in range(IMAGE_NUM)]

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            all_json[idx] = result
    
    json.dump(all_json, open(OUTPUT_DIR+'.json', 'w'), indent=4)
