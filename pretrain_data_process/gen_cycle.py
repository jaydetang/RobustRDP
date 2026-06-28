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


MOL_SIZE_RANGE = (100, 200)

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


def draw_curve_arrow(draw, start, end, color=(0,0,0), width=3, arrow_length=15, curvature=-0.2):
    x0, y0 = start
    x1, y1 = end

    # 计算中点
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2

    # 计算法向量（垂直方向）
    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy)
    if length == 0:
        length = 1e-8  # 防止除0
    ndx = -dy / length
    ndy = dx / length

    # 计算控制点（中点沿法向量偏移）
    control_x = mx + ndx * length * curvature
    control_y = my + ndy * length * curvature

    # 采样贝塞尔曲线的点
    points = []
    for t in [i/20 for i in range(21)]:  # 0到1，取21个点
        xt = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * control_x + t ** 2 * x1
        yt = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * control_y + t ** 2 * y1
        points.append((xt, yt))

    # 画曲线（PIL不支持真正的曲线，只能用多段线逼近）
    draw.line(points, fill=color, width=width)

    # 算最后一小段的方向（最后两个点）
    (x_prev, y_prev), (x_last, y_last) = points[-2], points[-1]
    dx = x_last - x_prev
    dy = y_last - y_prev
    length = math.hypot(dx, dy)
    if length == 0:
        length = 1e-8
    dx /= length
    dy /= length

    # 算箭头两边点
    arrow_angle = math.radians(20)  # 箭头展开角度
    sin_a = math.sin(arrow_angle)
    cos_a = math.cos(arrow_angle)

    lx = cos_a * dx - sin_a * dy
    ly = sin_a * dx + cos_a * dy
    rx = cos_a * dx + sin_a * dy
    ry = -sin_a * dx + cos_a * dy

    left_point = (x_last - arrow_length * lx, y_last - arrow_length * ly)
    right_point = (x_last - arrow_length * rx, y_last - arrow_length * ry)

    # 画箭头
    draw.line([left_point, (x_last, y_last)], fill=color, width=width)
    draw.line([right_point, (x_last, y_last)], fill=color, width=width)


# 随机生成反应条件文本
def random_condition_text():
    return random.sample(cond_list, 1)

# 生成单个反应方程
def generate_reaction():
    num_mols = 1
    mols = get_random_smiles(num_mols)
    
    num_conds = 1
    conds = []
    for i in range(num_conds):
        if random.random() < 0.5:  # 0%概率条件是文本
            condition = random_condition_text()[0]
            condition_is_text = True
        else:
            condition = get_random_smiles(1)[0]
            condition_is_text = False
        conds.append((condition, condition_is_text))
        
    return mols, conds


def arrange_mols(canvas, start_x, start_y, reactions):
    def func(canvas, mol_img, x, y):
        mol_w, mol_h = mol_img.size
        canvas.paste(mol_img, (x, y))
        bbox = [x, y, x+mol_w, y+mol_h]
        return [{"bbox": bbox, "content": "molecule"}]
    
    if len(reactions)==4:
        grid_pos = [[None]*2 for i in range(3)]
        x = start_x + random.randint(300, 400)
        y = start_y
        mol_img = draw_molecule(reactions[0][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[0][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(250, 300)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[1][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][2] + random.randint(50, 200)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[2][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][1] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] + random.randint(-100, 100)
        y = min(grid_pos[1][0][0]["bbox"][3], grid_pos[1][1][0]["bbox"][3]) + random.randint(150, 250)
        mol_img = draw_molecule(reactions[3][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[2][0] = func(canvas, mol_img, x, y)
    elif len(reactions)==5:
        grid_pos = [[None]*2 for i in range(3)]
        x = start_x + random.randint(300, 400)
        y = start_y
        mol_img = draw_molecule(reactions[0][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[0][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(250, 300)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[1][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][2] + random.randint(50, 200)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[2][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][1] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(150, 200)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(350, 450)
        mol_img = draw_molecule(reactions[3][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[2][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][2] + random.randint(20, 100)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(350, 450)
        mol_img = draw_molecule(reactions[2][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[2][1] = func(canvas, mol_img, x, y)
    elif len(reactions)==6:
        grid_pos = [[None]*2 for i in range(4)]
        x = start_x + random.randint(300, 400)
        y = start_y
        mol_img = draw_molecule(reactions[0][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[0][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(200, 350)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[1][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][2] + random.randint(50, 250)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(150, 250)
        mol_img = draw_molecule(reactions[2][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[1][1] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(200, 350)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(350, 450)
        mol_img = draw_molecule(reactions[3][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[2][0] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][2] + random.randint(50, 250)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(350, 450)
        mol_img = draw_molecule(reactions[4][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[2][1] = func(canvas, mol_img, x, y)

        x = grid_pos[0][0][0]["bbox"][0] - random.randint(10, 10)
        y = grid_pos[0][0][0]["bbox"][3] + random.randint(550, 650)
        mol_img = draw_molecule(reactions[5][0][0], random.randint(*MOL_SIZE_RANGE))
        grid_pos[3][0] = func(canvas, mol_img, x, y)

    return grid_pos


def arrange_arrows(canvas, grid_pos, reactions):
    def shorten(x1, y1, x2, y2, shortening_factor=0.8):
        dx = x2 - x1
        dy = y2 - y1

        x1_new = int(x1 + shortening_factor * dx)
        y1_new = int(y1 + shortening_factor * dy)
        x2_new = int(x2 - shortening_factor * dx)
        y2_new = int(y2 - shortening_factor * dy)

        return x2_new, y2_new, x1_new, y1_new

    def get_direc(x1, y1, x2, y2):
        dx = x2 - x1
        dy = y2 - y1

        if dx >= 0 and dy > 0:
            return 0  # 右下
        elif dx < 0 and dy >= 0:
            return 1  # 左下
        elif dx <= 0 and dy < 0:
            return 2  # 左上
        else:
            return 3  # 右上


    draw = ImageDraw.Draw(canvas)
    json_output = []
    arrow_color = (0, 0, 0)
    if len(reactions) == 4:
        mols_pos_list = [grid_pos[0][0][0], grid_pos[1][1][0], grid_pos[2][0][0], grid_pos[1][0][0]]
    elif len(reactions) == 5:
        mols_pos_list = [grid_pos[0][0][0], grid_pos[1][1][0], grid_pos[2][1][0], grid_pos[2][0][0], grid_pos[1][0][0]]
    elif len(reactions) == 6:
        mols_pos_list = [grid_pos[0][0][0], grid_pos[1][1][0], grid_pos[2][1][0], grid_pos[3][0][0], grid_pos[2][0][0], grid_pos[1][0][0]]

    for i in range(len(reactions)):
        reaction_entry = {"reactants": [], "conditions": [], "products": []}
        center_1 = ((mols_pos_list[i]['bbox'][0]+mols_pos_list[i]['bbox'][2])//2, (mols_pos_list[i]['bbox'][1]+mols_pos_list[i]['bbox'][3])//2)
        center_2 = ((mols_pos_list[(i+1)%len(reactions)]['bbox'][0]+mols_pos_list[(i+1)%len(reactions)]['bbox'][2])//2, (mols_pos_list[(i+1)%len(reactions)]['bbox'][1]+mols_pos_list[(i+1)%len(reactions)]['bbox'][3])//2)
        x1, y1, x2, y2 = shorten(center_1[0], center_1[1], center_2[0], center_2[1])
        arrow_start, arrow_end = (x1, y1), (x2, y2)
        arrow_center = ((x1+x2)//2, (y1+y2)//2)
        arrow_direc = get_direc(x1, y1, x2, y2)

        condition, condition_is_text = reactions[i][1][0]
        if condition_is_text:
            cond_img = draw_cond_img(condition, font_path_list)
        else:
            cond_img = draw_molecule(condition, random.randint(*MOL_SIZE_RANGE)*0.8)

        cond_w, cond_h = cond_img.size
        if arrow_direc == 0:
            cond_x, cond_y = arrow_center[0], arrow_center[1]-cond_h-random.randint(10, 15)
        elif arrow_direc == 1:
            cond_x, cond_y = arrow_center[0], arrow_center[1]+random.randint(10, 15)
        elif arrow_direc == 2:
            cond_x, cond_y = arrow_center[0]-cond_w-random.randint(10, 15), arrow_center[1]
        elif arrow_direc == 3:
            cond_x, cond_y = arrow_center[0]-cond_w-random.randint(10, 15), arrow_center[1]-cond_h-random.randint(10, 15)
        canvas.paste(cond_img, (cond_x, cond_y))
        cond_bbox = {"bbox": [cond_x, cond_y, cond_x+cond_w, cond_y+cond_h], "content": "text" if condition_is_text else "molecule"}
        # draw_arrow(draw, arrow_start, arrow_end, color=arrow_color)
        draw_curve_arrow(draw, arrow_start, arrow_end, color=arrow_color)

        reaction_entry["reactants"] = [mols_pos_list[i]]
        reaction_entry["conditions"] = [cond_bbox]
        reaction_entry["products"] = [mols_pos_list[(i+1)%len(reactions)]]
        json_output.append(reaction_entry)

    return json_output


# 布局函数（以单行为例，其他类似扩展）
def arrange_reactions(canvas, start_x, start_y, reactions):    
    grid_pos = arrange_mols(canvas, start_x, start_y, reactions)
    json_output = arrange_arrows(canvas, grid_pos, reactions)
        
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
    
    num_equations = random.randint(4, 6)
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
    OUTPUT_DIR = './pretrain_data_process/images/cycle'
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    PREFIX = 'cycle_{}'    

    all_json = [None]*IMAGE_NUM
    with ProcessPoolExecutor(max_workers=120) as executor:
        futures = [executor.submit(generate_image_and_json, OUTPUT_DIR, PREFIX.format(i)) for i in range(IMAGE_NUM)]

        for idx, future in enumerate(as_completed(futures)):
            result = future.result()
            all_json[idx] = result
    
    json.dump(all_json, open(OUTPUT_DIR+'.json', 'w'), indent=4)
