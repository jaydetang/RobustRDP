from PIL import Image
import os
import math
import json
from tqdm import tqdm

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

id_to_type = {1: "structure", 2: "text", 3: "identifier", 4: "supplement"}
target_size = 1000

def convert_to_reaction_parsing_format(reactions_data):
    """
    Convert reaction data to the special token format for reaction parsing
    """
    output_parts = []
    
    for reaction in reactions_data:
        reaction_parts = ["<rxn>"]
        
        # Process reactants
        reaction_parts.append("<rct>")
        for item in reaction["reactants"]:
            bbox = item["bbox"]
            content_type = "<mol>" if item["content"] == "molecule" else "<txt>"
            reaction_parts.append(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}{content_type}")
        
        # Process conditions
        reaction_parts.append("<cnd>")
        for item in reaction["conditions"]:
            bbox = item["bbox"]
            content_type = "<mol>" if item["content"] == "molecule" else "<txt>"
            reaction_parts.append(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}{content_type}")
        
        # Process products
        reaction_parts.append("<prd>")
        for item in reaction["products"]:
            bbox = item["bbox"]
            content_type = "<mol>" if item["content"] == "molecule" else "<txt>"
            reaction_parts.append(f"{bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}{content_type}")
        
        output_parts.append("".join(reaction_parts))
    
    return "\n".join(output_parts)


def process_original_data(data, name_to_scale, data_dir, output_images, target_size, mode):
    res_llm = []
    for index, item in tqdm(enumerate(data["images"])):
        image_name = item["file_name"]
        scale = name_to_scale[image_name]
        image_path = os.path.join(output_images, item["file_name"])
        w, h = Image.open(image_path).size
        assert w%28 == 0 and h%28 == 0 and w*h <= target_size*target_size

        bboxes = item["bboxes"]
        id_to_bbox = {}
        for box in bboxes:
            box["bbox"] = [box["bbox"][0]*scale[0], box["bbox"][1]*scale[1], box["bbox"][2]*scale[0], box["bbox"][3]*scale[1]]
            id_to_bbox[box["id"]] = box
        reactions = item["reactions"]
        cur_output = []
        
        try:
            for rxn in reactions:
                tmp = {"reactants": [],
                    "conditions": [],
                    "products": []}
                for role in ["reactants", "conditions", "products"]:
                    for b_id in rxn[role]:
                        bbox_tlx = id_to_bbox[b_id]["bbox"][0]
                        bbox_tly = id_to_bbox[b_id]["bbox"][1]
                        bbox_brx = id_to_bbox[b_id]["bbox"][0]+id_to_bbox[b_id]["bbox"][2]
                        bbox_bry = id_to_bbox[b_id]["bbox"][1]+id_to_bbox[b_id]["bbox"][3]
                        tmp[role].append(
                            {"bbox": [min(int(bbox_tlx), w),
                                    min(int(bbox_tly), h),
                                    min(int(bbox_brx), w),
                                    min(int(bbox_bry), h)
                                    ],
                            "content": "molecule" if id_to_bbox[b_id]["category_id"]==1 else "text"
                            })
                cur_output.append(tmp)
        except Exception as e:
            print("index: ", index, "item: ", item)
            exit()

        special_token_output = convert_to_reaction_parsing_format(cur_output)
        
        res_llm.append({
            "messages": [
                    {
                        "content": f"<image> {reaction_parsing_instruct}",
                        "role": "user"
                    },
                    {
                        "content": special_token_output, # json.dumps(cur_output) is used to convert the list of dictionaries to a JSON string
                        "role": "assistant"
                    },
                    ],
                    "images": [
                        image_path
                    ]
                })
    json.dump(res_llm, open(os.path.join(data_dir, f"{mode}_downsampled_llm.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)
    
    for index, item in tqdm(enumerate(data["images"])):
        image_name = item["file_name"]
        scale = name_to_scale[image_name]
        image_path = os.path.join(output_images, item["file_name"])
        item["file_name"] = image_path
        w, h = Image.open(image_path).size
        item["width"] = w
        item["height"] = h
        bboxes = item["bboxes"]
        for box in bboxes:
            for i in range(len(box["bbox"])):
                box["bbox"][i] = int(box["bbox"][i])
    json.dump(data, open(os.path.join(data_dir, f"{mode}_downsampled_rxn.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def downsample_image(image_path, output_path=None, save_image=True):
    """
    Downsample a high-resolution image so that both dimensions are multiples of 28 and do not exceed target_size
    
    Args:
        image_path (str): Input image path
        output_path (str): Output image path, appends '_downsampled' to the original filename if None
        save_image (bool): Whether to save the image to file
    
    Returns:
        numpy.ndarray: Downsampled image array
        tuple: (new width, new height)
    """
    
    # Read image
    if isinstance(image_path, str):
        # Read image using PIL
        img = Image.open(image_path)
        # Convert to RGB format (if RGBA or other formats)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        # If a numpy array is passed
        img = Image.fromarray(image_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
    
    original_width, original_height = img.size
    # print(f"Original image dimensions: {original_width} x {original_height}")

    
    max_possible_width = target_size // 28 * 28
    max_possible_height = target_size // 28 * 28
    # Calculate the scaling ratio while preserving the aspect ratio
    scale_x = min(max_possible_width / original_width, 1.0)
    scale_y = min(max_possible_height / original_height, 1.0)

    
    # Ensure the scaled area does not exceed target_size * target_size
    scale = math.sqrt(scale_x * scale_y)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Adjust to a multiple of 28 (rounding down)
    new_width = (new_width // 28) * 28
    new_height = (new_height // 28) * 28
        
    scale_x = new_width / original_width
    scale_y = new_height / original_height
    downsampled_image = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
    
    # Save image
    if save_image:
        if output_path is None:
            # Generate output path
            if isinstance(image_path, str):
                name, ext = os.path.splitext(image_path)
                output_path = f"{name}_downsampled{ext}"
            else:
                output_path = "downsampled_image.jpg"
        
        # Use PIL to save the image to ensure quality
        downsampled_image.save(output_path, quality=95)
    return (scale_x, scale_y), (new_width, new_height)


# Define a function to process a single image (for multiprocessing)
def process_single_image(args):
    input_path, output_path, filename = args
    try:
        scale, size = downsample_image(input_path, output_path)
        return filename, scale, None
    except Exception as e:
        return filename, None, str(e)


def multiprocess_downsample(input_dir, output_dir, num_processes=None):
    """
    Concurrently process all images in a folder using multiprocessing, no batching needed
    
    Args:
        input_dir (str): Input folder path
        output_dir (str): Output folder path
        num_processes (int): Number of processes, default is None (uses min(32, CPU cores))
    
    Returns:
        dict: Mapping from image filename to scale factor
    """
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    # Supported image formats
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Get all image files
    all_images = [f for f in os.listdir(input_dir) if f.lower().endswith(supported_formats)]
    total_images = len(all_images)
    print(f"Found a total of {total_images} images")
    
    if total_images == 0:
        print("Warning: No image files found")
        return {}
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Prepare task parameters
    tasks = []
    for filename in all_images:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        tasks.append((input_path, output_path, filename))
    
    # Set the number of processes
    if num_processes is None:
        cpu_count = multiprocessing.cpu_count()
        # Limit the maximum number of processes to avoid running out of memory
        num_processes = min(100, cpu_count)
    # Ensure the number of processes does not exceed the number of images
    num_processes = min(num_processes, total_images)
    print(f"Using {num_processes} processes for concurrent processing...")
    
    # Use ProcessPoolExecutor for multiprocessing
    name_to_scale = {}
    failed_images = []
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        # Submit all tasks
        future_to_filename = {executor.submit(process_single_image, task): task[2] for task in tasks}
        
        # Use tqdm to display progress
        for future in tqdm(as_completed(future_to_filename), total=len(tasks), desc="Processing images"):
            filename = future_to_filename[future]
            try:
                filename_result, scale, error = future.result()
                if error is None:
                    name_to_scale[filename_result] = scale
                else:
                    failed_images.append((filename_result, error))
            except Exception as e:
                failed_images.append((filename, str(e)))
    
    # Report failures
    if failed_images:
        print(f"\nWarning: {len(failed_images)} images failed to process:")
        for filename, error in failed_images:
            print(f"  {filename}: {error}")
    else:
        print("\nAll images processed successfully!")
    
    print(f"Processing completed, {len(name_to_scale)} succeeded, {len(failed_images)} failed")
    return name_to_scale


def down_sample_main(input_images, input_file, mode="train"):
    data_dir = os.path.dirname(input_images)
    output_images = input_images + "_resized"

    # process images
    name_to_scale = multiprocess_downsample(input_images, output_images)

    # process json
    train_data = json.load(open(input_file, 'r', encoding='utf-8'))
    process_original_data(train_data, name_to_scale, data_dir, output_images, target_size, mode)
