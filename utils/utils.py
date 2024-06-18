import os
import gc
import torch
from config import *

output_filtering = lambda x, model: x.split(model.prompt_rule["test_start"])[-1].split(model.prompt_rule["test_end"])[0].strip()

def memory_optimization():
    # memory deallocation
    gc.collect()

    # removing cache
    torch.cuda.empty_cache()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        assert False

def freeze_model(model):
    for param in model.parameters():
        param.requires_grad=False

def switching_model(model, updating_param):
    if updating_param == 'all':
        for name, param in model.named_parameters():
            param.requires_grad=True
        return

    for name, param in model.named_parameters():
        if 'float' in str(param.dtype):
            if sum([up_param in name for up_param in updating_param]):
                param.requires_grad=True
            else:
                param.requires_grad=False

def weight_upload(tensor_dict, model):
    used_name = []
    for name, param in tensor_dict.items():
        split_name = name.split('.')
        
        traversal = model
        for module_name in split_name:
            traversal = getattr(traversal, module_name)
        # logging
        # print(f'{name}: {(traversal==param.to(traversal.device)).sum()}/{(traversal!=param.to(traversal.device)).sum()}')
        setattr(traversal, 'data', param.to(traversal.device))
        used_name.append(name)

    for name in used_name:
        del tensor_dict[name]

def find_special_token(string, special_token):
    start = 0
    while True:
        start = string.find(special_token, start)
        if start == -1: return
        yield start
        start += len(special_token) # use start += 1 to find overlapping matches

def add_bundle_tokens(input_string, special_token, num):

    # number of special tokens in input_string
    num_special_tokens = len(list(find_special_token(input_string, special_token)))

    # No special token -> return the raw
    if not num_special_tokens:
        return input_string
    
    result = ""
    index = 0
    while index < len(input_string):
        if input_string[index:index + len(special_token)] == special_token:
            result += special_token * num
            index += len(special_token)
        else:
            result += input_string[index]
            index += 1

    assert len(list(find_special_token(result, special_token))) == num_special_tokens * num
    return result

def make_instruction(question, dataset, prompt_rule):
    system_prompt = make_human_string("You are AI model created by Byung-Kwan Lee, Ph.D. candidate, KAIST EE, of which AI model name is TroL (Traversal of Layers).",
                                      "You must give helpful, detailed, and polite answers to the user's questions",
                                      split=' ')
    
    if dataset != "mmmu" and dataset != "mathverse" and dataset != "hallusionbench" and dataset != "demo":
        question = "<image>" + question

    if dataset in ["sqa", "mmbench", "mmbench_cn", "mmbench_dev", "mmbench_cn_dev", "seed", "qbench", "ai2d", "mmstar"]:
        question = question + "\nAnswer with the option's letter from the given choices directly."

    elif dataset in ["vqav2", "gqa", "pope", "chartqa"]:
        question = question + "\nAnswer the question using a single word or phrase."

    elif dataset in ["vizwiz"]:
        question = question + "\nWhen the provided information is insufficient, respond with 'Unanswerable'. Answer the question using a single word or phrase."

    elif dataset in ["mmmu"]:
        if "A." in question:
            question = question + "\nAnswer with the option's letter from the given choices directly."
        else:
            question = question + "\nAnswer the question using a single word or phrase."
        
    elif dataset in ["hallusionbench"]:
        if "Please answer yes or no." not in question:
            question = question + "\nPlease answer yes or no."
    
    qa_prompt = make_human_string(prompt_rule["system_start"]+system_prompt+prompt_rule["system_end"],
                                  prompt_rule["user_start"]+question+prompt_rule["user_end"],
                                  prompt_rule["assistant_start"],
                                  split=prompt_rule["split"])

    return qa_prompt

def make_human_string(*args, split):
    out = ''
    for i, arg in enumerate(args):
        out += arg
        if i != len(args)-1:
            out += split
    return out

def get_max_new_tokens(data_name):
    if data_name.lower() in ["mme", "pope", "sqa", "mmbench", "mmbench_cn", "mmbench_dev","mmbench_cn_dev", "seed", "qbench", "ai2d", "mmstar", "vqav2", "gqa", "chartqa", "hallusionbench", "textvqa", "mmmu"]:
        return 5
    if data_name.lower() in ["llava", "mm-vet"]:
        return 1024
    else:
        return 512

def pixel_shuffle(x, scale_factor=0.5):
    n, w, h, c = x.size()
    # N, W, H, C --> N, W, H * scale, C // scale
    x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
    # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
    x = x.permute(0, 2, 1, 3).contiguous()
    # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
    x = x.view(n, int(h * scale_factor), int(w * scale_factor),
                int(c / (scale_factor * scale_factor)))
    x = x.permute(0, 2, 1, 3).contiguous()
    return x

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
dynamic_transform = build_transform(input_size=448)

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
    from torchvision.transforms.functional import to_pil_image
    image = to_pil_image(image)
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images