import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from easydict import EasyDict as edict
import subprocess
import os
import json
import numpy as np

def get_winoground_scores(scores):
    # scores can be a tuple of (scores_t2i, scores_i2t) or just scores_t2i
    # The size should be N x N_image x N_text for both scores_t2i and scores_i2t
    # For ITC/ITM like models, these matrix should be symmetric
    if isinstance(scores, tuple):
        # Note both are N x N_image x N_text
        scores_t2i = scores[0]
        scores_i2t = scores[1]
    else:
        scores_t2i = scores
        scores_i2t = scores
    ids = list(range(scores_t2i.shape[0]))
    winoground_scores = []
    for id, score_i2t, score_t2i in zip(ids, scores_i2t, scores_t2i):
        winoground_scores.append({
            'i2t' : {
                "id" : id,
                "c0_i0": score_i2t[0][0],
                "c0_i1": score_i2t[1][0],
                "c1_i0": score_i2t[0][1],
                "c1_i1": score_i2t[1][1]},
            't2i' : {
                "id" : id,
                "c0_i0": score_t2i[0][0],
                "c0_i1": score_t2i[1][0],
                "c1_i0": score_t2i[0][1],
                "c1_i1": score_t2i[1][1]},
            })
    return winoground_scores


def get_winoground_acc(scores):
    text_correct_count = 0
    image_correct_count = 0
    group_correct_count = 0
    def text_correct(result):
        return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

    def image_correct(result):
        return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

    def group_correct(result):
        return image_correct(result['t2i']) and text_correct(result['i2t'])
    
    for result in scores:
        text_correct_count += 1 if text_correct(result['i2t']) else 0
        image_correct_count += 1 if image_correct(result['t2i']) else 0
        group_correct_count += 1 if group_correct(result) else 0

    denominator = len(scores)
    result = {
            'text': round(text_correct_count*100/denominator,4),
            'image': round(image_correct_count*100/denominator,4),
            'group': round(group_correct_count*100/denominator,4),
        }
    return result

# def get_winoground_acc(scores, saved_path="./clip_l14.json", save=False, demonimator=False):
#     ''' Accuracy for winoground'''
#     text_correct_count = 0
#     image_correct_count = 0
#     group_correct_count = 0
#     text_corrects = []
#     image_corrects = []
#     group_corrects = []
#     def text_correct(result):
#         return result["c0_i0"] > result["c1_i0"] and result["c1_i1"] > result["c0_i1"]

#     def image_correct(result):
#         return result["c0_i0"] > result["c0_i1"] and result["c1_i1"] > result["c1_i0"]

#     def group_correct(result):
#         return image_correct(result['t2i']) and text_correct(result['i2t'])
    
#     for idx, result in enumerate(scores):
#         text_correct_count += 1 if text_correct(result['i2t']) else 0
#         image_correct_count += 1 if image_correct(result['t2i']) else 0
#         group_correct_count += 1 if group_correct(result) else 0
#         if text_correct(result['i2t']):
#             text_corrects.append(idx)
#         if image_correct(result['t2i']):
#             image_corrects.append(idx)
#         if group_correct(result):
#             group_corrects.append(idx)
#     denominator = len(scores)
#     if save:
#         import json
#         with open(saved_path, "w") as f:
#             json.dump({
#                 'text': text_corrects,
#                 'image': image_corrects,
#                 'group': group_corrects,
#                 'denominator': denominator,
#             }, f, indent=2)
#     if demonimator:
#         result = {
#             'text': round(text_correct_count*100/denominator,4),
#             'image': round(image_correct_count*100/denominator,4),
#             'group': round(group_correct_count*100/denominator,4),
#         }
#     else:
#         result = {
#             'text': text_correct_count,
#             'image': image_correct_count,
#             'group': group_correct_count,
#             'denominator': denominator,
#         }
#     return result


def get_winoground_analysis(scores):
    ''' Error analysis for winoground'''
    text_flip_count = 0
    image_flip_count = 0
    text_same_count = 0
    image_same_count = 0
    def text_flip(result):
        # If both images prefer the wrong caption
        return result["c0_i0"] < result["c1_i0"] and result["c1_i1"] < result["c0_i1"]

    def text_same(result):
        # If both images prefer the same caption
        return (result["c0_i0"] > result["c1_i0"] and result["c1_i1"] < result["c0_i1"]) or \
            (result["c0_i0"] < result["c1_i0"] and result["c1_i1"] > result["c0_i1"])
    
    def image_flip(result):
        # It both captions prefer the wrong image
        return result["c0_i0"] < result["c0_i1"] and result["c1_i1"] < result["c1_i0"]
    
    def image_same(result):
        # If both captions prefer the same image
        return (result["c0_i0"] > result["c0_i1"] and result["c1_i1"] < result["c1_i0"]) or \
            (result["c0_i0"] < result["c0_i1"] and result["c1_i1"] > result["c1_i0"])

    for result in scores:
        text_flip_count += 1 if text_flip(result['i2t']) else 0
        image_flip_count += 1 if image_flip(result['t2i']) else 0
        text_same_count += 1 if text_same(result['i2t']) else 0
        image_same_count += 1 if image_same(result['t2i']) else 0

    denominator = len(scores)
    # result = {
    #     'text_flip': text_flip_count/denominator,
    #     'image_flip': text_flip_count/denominator,
    #     'text_same': text_same_count/denominator,
    #     'image_same': image_same_count/denominator,
    # }
    result = {
        'text_flip': text_flip_count,
        'image_flip': text_flip_count,
        'text_same': text_same_count,
        'image_same': image_same_count,
        'denominator': denominator,
    }
    return result


class Winoground(Dataset):
    def __init__(self, root_dir='./', transform=None, return_image_paths=False):
        """Winoground dataset.
        Args:
            root_dir (string): Directory where the dataset will be stored.
            transform (callable, optional): Optional image transform to be applied
            return_image_paths (bool): If True, return image paths (str) instead of Tensors
        """
        self.root_dir = os.path.join(root_dir, "winoground")
        if not os.path.exists(self.root_dir):
            subprocess.call(["gdown", "--id", "1Lril_90vjsbL_2qOaxMu3I-aPpckCDiF", "--output", os.path.join(root_dir, "winoground.zip")])
            subprocess.call(["unzip", "winoground.zip"], cwd=root_dir)
            subprocess.call(["rm", "winoground.zip"], cwd=root_dir)
        csv_file = os.path.join(self.root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(csv_file).to_dict(orient='records')
        self.transform = transform
        self.original_tags = self.get_original_tags()
        self.new_tags = self.get_new_tags()
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
            
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        assert self.metadata[idx]['id'] == idx
        image_0_path = os.path.join(self.root_dir, self.metadata[idx]['image_0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[idx]['image_1'])
        caption_0 = self.metadata[idx]['caption_0']
        caption_1 = self.metadata[idx]['caption_1']
        
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            image_0 = Image.open(image_0_path)
            image_1 = Image.open(image_1_path)
        
        if self.transform:
            image_0 = self.transform(image_0)
            image_1 = self.transform(image_1)

        sample = edict({"id": idx,
                        "image_options": [image_0, image_1],
                        "caption_options": [caption_0, caption_1]})

        return sample
    
    def get_original_tags(self, path='examples.jsonl'):
        tags = {}
        file_path = os.path.join(self.root_dir, path)

        # BUG: winoground_dict = json.load(open(os.path.join(self.root_dir, path)))
        # FIX: Open the .jsonl file and parse it line by line
        winoground_list = []
        with open(file_path, 'r') as file:
            for line in file:
                # Parse the JSON object in each line and append to the list
                json_object = json.loads(line)
                winoground_list.append(json_object)
        
        for example in winoground_list:
            if example['num_main_preds'] == 1:
                if '1 Main Pred' not in tags:
                    tags["1 Main Pred"] = []
                tags['1 Main Pred'].append(example["id"])
            elif example['num_main_preds'] == 2:
                if '2 Main Pred' not in tags:
                    tags["2 Main Pred"] = []
                tags['2 Main Pred'].append(example["id"])
            else:
                # This won't happen
                raise ValueError(f"num_main_preds: {example['num_main_preds']}")
            if example["collapsed_tag"] not in tags:
                tags[example["collapsed_tag"]] = []
            tags[example["collapsed_tag"]].append(example["id"])
        return tags

    def get_new_tags(self, path="why_winoground_hard.json"):
        new_tag_dict = json.load(open(os.path.join(self.root_dir, path)))
        tags = {}
        for idx in new_tag_dict:
            curr_tags = new_tag_dict[idx]
            if len(curr_tags) == 0:
                if "No Tag" not in tags:
                    tags["No Tag"] = []
                tags["No Tag"].append(int(idx))
            for tag in curr_tags:
                if tag not in tags:
                    tags[tag] = []
                tags[tag].append(int(idx))
        return tags
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores, save=True)
        print("Winoground text score:", acc['text'])
        print("Winoground image score:", acc['image'])
        print("Winoground group score:", acc['group'])
        
        error = get_winoground_analysis(winoground_scores)
        print("Winoground text flip:", error['text_flip'])
        print("Winoground text same:", error['text_same'])
        print("Winoground image flip:", error['image_flip'])
        print("Winoground image same:", error['image_same'])
        
        results = {}
        results['all'] = acc
        for tag in self.original_tags:
            results[tag] = get_winoground_acc([winoground_scores[i] for i in self.original_tags[tag]])
            print(f"Winoground {tag} text score: {results[tag]['text']}")
            print(f"Winoground {tag} image score: {results[tag]['image']}")
            print(f"Winoground {tag} group score: {results[tag]['group']}")
        for tag in self.new_tags:
            results[tag] = get_winoground_acc([winoground_scores[i] for i in self.new_tags[tag]])
            print(f"Winoground {tag} text score: {results[tag]['text']}")
            print(f"Winoground {tag} image score: {results[tag]['image']}")
            print(f"Winoground {tag} group score: {results[tag]['group']}")
        return results, acc['group']
    
    
class EqBen_Val(Dataset):
    def __init__(self, transform=None, root_dir='./', return_image_paths=False):
        self.root_dir = os.path.join(root_dir, "eqben_val")
        if not os.path.exists(self.root_dir):
            subprocess.call(["gdown", "--id", "1XGbvdVcQjIjVfbcihvjh8ovwUtuPEbsW", "--output", os.path.join(root_dir, "eqben_val.zip")])
            subprocess.call(["unzip", "eqben_val.zip"], cwd=root_dir)
            subprocess.call(["rm", "eqben_val.zip"], cwd=root_dir)
        csv_file = os.path.join(self.root_dir, 'metadata.csv')
        self.metadata = pd.read_csv(csv_file).to_dict(orient='records')
        self.transform = transform
        self.return_image_paths = return_image_paths
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
    
    def process_img(self, image_path):
        if self.return_image_paths:
            return image_path
        file_type = image_path.split('.')[-1]
        if file_type == 'npy':
            image = self.process_img_npy(image_path)
        else:
            image = self.process_img_pixel(image_path)
        return image

    def process_img_pixel(self, image_path):
        # Define the model-specifc data pre-process (e.g., augmentation) for image pixel.
        image = Image.open(image_path).convert("RGB")
        return image

    def process_img_npy(self, image_path):
        # Define the model-specifc data pre-process (e.g., augmentation) for image numpy file (Youcook2).
        image = Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
        return image
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        assert self.metadata[idx]['id'] == idx
        image_0_path = os.path.join(self.root_dir, self.metadata[idx]['image_0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[idx]['image_1'])
        image_0 = self.process_img(image_0_path)
        image_1 = self.process_img(image_1_path)
        if self.transform:
            image_0 = self.transform(image_0)
            image_1 = self.transform(image_1)
        caption_0 = self.metadata[idx]['caption_0']
        caption_1 = self.metadata[idx]['caption_1']
        item = edict({"image_options": [image_0, image_1], "caption_options": [caption_0, caption_1]})
        return item
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("EQBen_Val text score:", acc['text'])
        print("EQBen_Val image score:", acc['image'])
        print("EQBen_Val group score:", acc['group'])
        error = get_winoground_analysis(winoground_scores)
        print("EQBen_Val text flip:", error['text_flip'])
        print("EQBen_Val text same:", error['text_same'])
        print("EQBen_Val image flip:", error['image_flip'])
        print("EQBen_Val image same:", error['image_same'])
        
        results = {}
        results['all'] = acc
        error = get_winoground_analysis(winoground_scores)
        print("EQBen_Val text flip:", error['text_flip'])
        print("EQBen_Val text same:", error['text_same'])
        print("EQBen_Val image flip:", error['image_flip'])
        print("EQBen_Val image same:", error['image_same'])
        return results, acc['group']
    
class EqBen_Mini(Dataset):
    def __init__(self, image_preprocess=None, root_dir='./', download=False, return_image_paths=True):
        self.preprocess = image_preprocess
        
        self.root_dir = os.path.join(root_dir, "eqben_vllm")
        if not os.path.exists(self.root_dir):
            # https://drive.google.com/file/d/11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM/view?usp=sharing
            os.makedirs(self.root_dir, exist_ok=True)
            subprocess.call(["gdown", "--id", "11YUTf06uzRHtFV8rYi96z4vTPi8_GNEM", "--output", os.path.join(self.root_dir, "eqben_vllm.zip")])
            subprocess.call(["unzip", "eqben_vllm.zip"], cwd=self.root_dir)
            
        self.root_dir = os.path.join(root_dir, "eqben_vllm", "images")
        self.subset_types = {
            'eqbensd': ['eqbensd'],
            'eqbenk': ['eqbenkubric_cnt', 'eqbenkubric_loc', 'eqbenkubric_attr'],
            'eqbeng': ['eqbengebc'],
            'eqbenag': ['eqbenag'],
            'eqbeny': ['eqbenyoucook2'],
        }
        json_file = os.path.join(root_dir, "eqben_vllm", "all_select.json")
        self.metadata = json.load(open(json_file, 'r'))
        self.subset_indices = {subset_type: [] for subset_type in self.subset_types}
        for item_idx, item in enumerate(self.metadata):
            image_path = item['image0']
            for subset_type in self.subset_types:
                if image_path.split('/')[0] in self.subset_types[subset_type]:
                    self.subset_indices[subset_type].append(item_idx)
                    break
        
        self.return_image_paths = return_image_paths
        self.transform = image_preprocess
        if self.return_image_paths:
            assert self.transform is None, "Cannot return image paths and apply transforms"
     
    def __len__(self):
        return len(self.metadata)
    
    def get_image_loader(self):
        def image_loader(image_path):
            if image_path.split('.')[-1] == 'npy':
                return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
            else:
                return Image.open(image_path).convert("RGB")
        return image_loader

    def image_loader(self, image_path):
        if image_path.split('.')[-1] == 'npy':
            return Image.fromarray(np.load(image_path)[:, :, [2, 1, 0]], 'RGB')
        else:
            return Image.open(image_path).convert("RGB")
    
    def __getitem__(self, index):
        image_0_path = os.path.join(self.root_dir, self.metadata[index]['image0'])
        image_1_path = os.path.join(self.root_dir, self.metadata[index]['image1'])
        if self.return_image_paths:
            image_0 = image_0_path
            image_1 = image_1_path
        else:
            if self.transform:
                image_0 = self.transform(self.image_loader(image_0_path))
                image_1 = self.transform(self.image_loader(image_1_path))
            else:
                image_0 = self.image_loader(image_0_path)
                image_1 = self.image_loader(image_1_path)
        
        caption_0 = self.metadata[index]['caption0']
        caption_1 = self.metadata[index]['caption1']
        item = edict({"image_options": [image_0, image_1], "caption_options": [caption_0, caption_1]})
        return item
    
    def evaluate_scores(self, scores):
        winoground_scores = get_winoground_scores(scores)
        acc = get_winoground_acc(winoground_scores)
        print("EQBen_Mini performance (overall)")
        print(f"{'Dataset': <70} {'Text': <10} {'Image': <10} {'Group': <10}")
        print(f"{'EQBen_Mini': <70} {acc['text']: <10.2f} {acc['image']: <10.2f} {acc['group']: <10.2f}")
        results = {}
        results['all'] = acc
        for subset_type in self.subset_types:
            subset_indices = self.subset_indices[subset_type]
            subset_scores = [winoground_scores[idx] for idx in subset_indices]
            subset_acc = get_winoground_acc(subset_scores)
            print(f"{'EQBen_Mini ' + subset_type: <70} {subset_acc['text']: <10.2f} {subset_acc['image']: <10.2f} {subset_acc['group']: <10.2f}")
            results[subset_type] = subset_acc
        return results, acc['group']