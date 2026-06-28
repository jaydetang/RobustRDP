import os
import json
import copy
import random
import argparse
import threading
import numpy as np
import torch
import traceback
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from utils import transforms as T
from utils.tokenizer import get_tokenizer
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


def gen_real_vrp_data(input_folder, output_filename):
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
current_idx = 5000

class ReactionDataset(Dataset):
    def __init__(self, args, tokenizer, data_file):
        super().__init__()
        self.args = args
        self.tokenizer = tokenizer

        with open(data_file) as f:
            self.data = json.load(f)['images']

        self.image_path = args.image_path
        self.format = args.format
        self.transform = make_transforms(args.rotate_augment)

    def __len__(self):
        return len(self.data)

    @property
    def pad_id(self):
        return self.tokenizer[self.format].PAD_ID

    def generate_sample(self, image, target):
        ref = {}
        # coordinates are normalized after transform
        image, target = self.transform(image, target)
        target['scale'] = 1.0
        
        self.save_image_and_target(image, target)  # Save augmented image and target to augment_data and augment_image_path
        
        ref['scale'] = target['scale']
        return image, ref

    def __getitem__(self, idx):
        image, target = self.load_and_prepare(idx)
        target['all_image_id'] = [target['image_id'].item()]
        if self.args.composite_augment:
            cnt = 0
            while idx % 2 == random.randrange(2) and cnt < 4:
                # Augment with probability 0.5
                n = len(self)
                idx2 = (idx + random.randrange(n)) % n
                image2, target2 = self.load_and_prepare(idx2)
                
                image, target = self.concat(image, target, image2, target2)
                target['all_image_id'].append(target2['image_id'].item())
                cnt += 1
        
        image, ref = self.generate_sample(image, target)
        ref['file_name'] = self.data[idx]['file_name']
        return [[idx, image, ref]]

    def load_and_prepare(self, idx):
        target = self.data[idx]
        path = os.path.join(self.image_path, target['file_name'])
        if not os.path.exists(path):
            print(path, "doesn't exists.", flush=True)
        image = Image.open(path).convert("RGB")
        image, target = self.prepare(image, target)
        return image, target

    def save_image_and_target(self, image, target):
        """Restore the image and target processed by prepare() back to the original dataset format and save"""
        global current_idx

        with data_lock:  # Ensure ID auto-increment and list append are atomic
            local_idx = current_idx
            current_idx += 1

        try:
            # 1. Reconstruct bboxes format in target
            # After the prepare() function, the original bboxes info has been converted to boxes, labels, etc., need to reconstruct
            bboxes = []
            if 'boxes' in target and 'labels' in target:
                boxes = target['boxes']
                labels = target['labels']
                areas = target.get('area', [])

                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.tolist()
                    # Convert back to [x, y, width, height] format
                    width = x2 - x1
                    height = y2 - y1

                    bbox_info = {
                        "id": i,
                        "bbox": [x1, y1, width, height],
                        "category_id": int(labels[i]) if i < len(labels) else 1,
                        }
                    bboxes.append(bbox_info)

            else:
                print("Warning: boxes or labels field missing in target", flush=True)
                print(f"Available fields: {list(target.keys())}", flush=True)
                # If no boxes info, at least create an empty bboxes list
                bboxes = []
                print("Creating empty bboxes list", flush=True)
            
            # 2. Save image
            # If image is a tensor, convert to PIL image
            if isinstance(image, torch.Tensor):
                # Denormalize
                mean = torch.tensor([0.485, 0.456, 0.406])
                std = torch.tensor([0.229, 0.224, 0.225])
                image_tensor = image.clone()
                for t, m, s in zip(image_tensor, mean, std):
                    t.mul_(s).add_(m)
                image_tensor = torch.clamp(image_tensor, 0, 1)
                # Convert to PIL image
                pil_image = TF.to_pil_image(image_tensor)
            else:
                pil_image = image
            
            # Generate augmented image filename
            all_ids_str = "_".join([str(id) for id in target.get('all_image_id', [])])
            aug_filename = f"aug_{local_idx:04d}_{all_ids_str}.png"

            if not os.path.exists(os.path.join(self.args.augmented_save_path, "images_aug_train")):
                os.makedirs(os.path.join(self.args.augmented_save_path, "images_aug_train"))
            aug_image_path = os.path.join(self.args.augmented_save_path, "images_aug_train", aug_filename)
            
            # Save image
            pil_image.save(aug_image_path)

            # Construct data item (using original dimensions)
            img_width, img_height = target.get('orig_size', [0, 0]).tolist()
            data_item = {
                "id": local_idx,
                "width": img_width,
                "height": img_height,
                "file_name": aug_filename,
                "license": 1,
                "bboxes": bboxes,
                "reactions": target.get('reactions', []),
                "corefs": None,
                "diagram_type": "graph",
                "source_image_ids": target.get('all_image_id', []),
                "is_composite": len(target.get('all_image_id', [])) > 1
            }
            
            # Add to augment_data
            with data_lock:
                augment_data["images"].append(data_item)
                
        except Exception as e:
            print(f"Error saving augmented data: {e}")
            traceback.print_exc()

    @staticmethod
    def save_augmented_dataset(args):
        """Save the complete augmented dataset to a JSON file"""
        save_path = os.path.join(args.augmented_save_path, "merged_aug_train_output.json")
        try:
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            
            augment_data["images"].sort(key=lambda x: x["id"])
            with open(save_path, 'w') as f:
                json.dump(augment_data, f, indent=2)
            
            print(f"Augmented dataset saved to: {save_path}")
            print(f"Total includes {len(augment_data['images'])} augmented samples")
            
        except Exception as e:
            print(f"Error saving augmented dataset: {e}")
    
    # Process the image and target so that they meet the model's input requirements
    def prepare(self, image, target):
        w, h = target['width'], target['height']

        image_id = target["id"]
        image_id = torch.tensor([image_id])

        anno = target["bboxes"]

        boxes = [obj['bbox'] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)
        

        target = copy.deepcopy(target)
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id
        target["orig_size"] = torch.as_tensor([int(w), int(h)])  # Save original dimensions [width, height]
        
        area = torch.tensor([obj["bbox"][2] * obj['bbox'][3] for obj in anno])
        target["area"] = area
        
        return image, target

    def concat(self, image1, target1, image2, target2):
        color = (255, 255, 255)
        if random.random() < 1:
            # Vertically concat two images
            w = max(image1.width, image2.width)
            h = image1.height + image2.height
            if image1.width > image2.width:
                x1, y1 = 0, 0
                x2, y2 = random.randint(0, image1.width - image2.width), image1.height
            else:
                x1, y1 = random.randint(0, image2.width - image1.width), 0
                x2, y2 = 0, image1.height
        else:
            # Horizontally concat two images
            w = image1.width + image2.width
            h = max(image1.height, image2.height)
            if image1.height > image2.height:
                x1, y1 = 0, 0
                x2, y2 = image1.width, random.randint(0, image1.height - image2.height)
            else:
                x1, y1 = 0, random.randint(0, image2.height - image1.height)
                x2, y2 = image1.width, 0
                
        image = Image.new('RGB', (w, h), color)
        image.paste(image1, (x1, y1))
        image.paste(image2, (x2, y2))
        
        target = {
            "image_id": target1["image_id"],
            "all_image_id": target1["all_image_id"],
            "orig_size": torch.as_tensor([int(w), int(h)]),
            "size": torch.as_tensor([int(w), int(h)])
        }
        target1["boxes"][:, 0::2] += x1
        target1["boxes"][:, 1::2] += y1
        target2["boxes"][:, 0::2] += x2
        target2["boxes"][:, 1::2] += y2
        for key in ["boxes", "labels", "area"]:
            target[key] = torch.cat([target1[key], target2[key]], dim=0)
        if "reactions" in target1:
            target["reactions"] = [r for r in target1["reactions"]]
            nbox = len(target1["boxes"])
            
            for r in target2["reactions"]:
                newr = {}
                for key, seq in r.items():
                    newr[key] = [x + nbox for x in seq]
                target["reactions"].append(newr)

        return image, target


def make_transforms(rotate_augment=False):
    if rotate_augment:
        return T.Compose([
            T.RandomRotate(),
            T.RandomHorizontalFlip(),
        ])
    else:
        return T.Compose([
            T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
        ])


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pix2seq', action='store_true')
    parser.add_argument('--coord_bins', type=int, default=100)
    parser.add_argument('--sep_xy', action='store_true')
    # Data paths
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--image_file', type=str, default=None)
    # Data format and augmentation
    parser.add_argument('--format', type=str, default='reaction')
    parser.add_argument('--aug_d', type=int, default=15)
    parser.add_argument('--rotate_augment', action='store_true')
    parser.add_argument('--composite_augment', action='store_true')
    # Save path
    parser.add_argument('--augmented_save_path', type=str, default='./sft_data_process/aug_data', help='Path to save augmented data')
    return parser.parse_args()


def gen_aug_vrp_data(args=None):
    if args is None:
        args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    torch.cuda.empty_cache()

    tokenizer = get_tokenizer(args)

    dm = ReactionDataset(args, tokenizer, args.image_file)

    def process_idx(i):
        return dm[i]
    
    max_workers = 100

    for j in range(args.aug_d):
        print(f"Pass {j}")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(tqdm(executor.map(process_idx, range(len(dm))), total=len(dm)))

    dm.save_augmented_dataset(args)


if __name__ == "__main__":
    # Convert raw data from RxnLabel to format for Multi-task SFT, obtaining 4240 real samples
    # sft_data_process/raw_data/train_downsampled_llm.json constitutes Vanilla Reaction Parsing SFT data
    gen_real_vrp_data('./sft_data_process/raw_data/labels_train', 
                    './sft_data_process/raw_data/merged_train_output.json')
    
    down_sample_main("./sft_data_process/raw_data/images_train", './sft_data_process/raw_data/merged_train_output.json', mode="train")

    # Apply data augmentation to raw data from RxnLabel, obtaining 63600 aug. samples for Multi-task SFT
    # sft_data_process/aug_data/train_downsampled_llm.json constitutes Vanilla Reaction Parsing SFT data
    aug_args = argparse.Namespace(
        image_path="./sft_data_process/raw_data/images_train",
        image_file="./sft_data_process/raw_data/merged_train_output.json",
        augmented_save_path="./sft_data_process/aug_data",
        format="reaction",
        aug_d=15,
        rotate_augment=True,
        composite_augment=True,
        seed=880,
        pix2seq=False,
        coord_bins=100,
        sep_xy=False,
    )
    gen_aug_vrp_data(aug_args)

    down_sample_main("./sft_data_process/aug_data/images_aug_train", "./sft_data_process/aug_data/merged_aug_train_output.json", mode="train")