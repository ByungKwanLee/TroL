import os
import torch
import argparse
import base64
from config import *
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from utils.utils import *
from datetime import timedelta
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trol.load_trol import load_trol
from eval.create_evaluator import Evaluator
from loader.create_eval_dataset import CreateEvalDataset
from accelerate import Accelerator, InitProcessGroupKwargs
from torchvision.transforms.functional import pil_to_tensor

class EvalDataset(CreateEvalDataset):
    def __init__(self, resol):
        super().__init__()

        # resolution
        self.resol = resol

        # select dataset
        self.eval_dataset = None

    def __getitem__(self, index):
        # img_path
        if 'image' in self.eval_dataset[index]:
            img_path = self.eval_dataset[index]['image']

            if img_path == "":
                # self.eval_dataset[index].update({'image': None})
                del self.eval_dataset[index]['image']
                return self.eval_dataset[index]
        else:
            # in case of multiple images like MMMU
            img_paths = self.eval_dataset[index]['images']
            images = [Image.open(BytesIO(img)).convert("RGB") for img in img_paths]
            img_tensors = [pil_to_tensor(img) for img in images]
            resized_img_tensors = [F.interpolate(img.unsqueeze(0), size=(self.resol, self.resol), mode='bicubic').squeeze(0) if self.resol else img for img in img_tensors]
            concat_img = torch.stack(resized_img_tensors)
            self.eval_dataset[index].update({'image': concat_img})
            return self.eval_dataset[index]
        
        # img may contain encoded data
        try:
            image = Image.open(os.path.join(DATASET_ROOT, img_path)).convert("RGB")
        except:
            try: 
                # correct file names for hallusionbench
                if img_path.find('png') != -1:
                    new_img_path = img_path.replace('png', 'PNG')
                else:
                    new_img_path = img_path.replace('PNG', 'png')
                image = Image.open(os.path.join(DATASET_ROOT, new_img_path)).convert("RGB")
            except:
                try : 
                    image = Image.open(BytesIO(base64.b64decode(img_path))).convert("RGB")
                except :
                    image = Image.open(BytesIO(img_path)).convert("RGB")
        
        img_tensor = pil_to_tensor(image)
        resized_img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.resol, self.resol), mode='bicubic').squeeze(0) if self.resol else img_tensor
        self.eval_dataset[index].update({'image': resized_img_tensor})

        return self.eval_dataset[index]

    def __len__(self):
        return len(self.eval_dataset)
    
    def update_dataset(self, dataset):
        self.eval_dataset = self.data[dataset]
    
def test(args):
    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator(kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=18000))])

    # loading model
    model, tokenizer = load_trol(link=args.link)

    # cpu -> gpu
    for param in model.parameters():
        if not param.is_cuda:
            param.data = param.to(accel.device)

    # Select datasets to eval
    if args.dataset[0] == "all":
        eval_datasets = EVAL_DATASETS
    else:
        eval_datasets = [args.dataset]

    # Initialize dataset & evaluator
    test_dataset = EvalDataset(resol=args.resol)
    evaluator = Evaluator()
    results = {}

    for data in eval_datasets:
        # Update dataset & evaluator
        evaluator.reset()
        test_dataset.update_dataset(dataset=data)
        test_dataloader = DataLoader(test_dataset, 
                                    batch_size=args.batch_size, 
                                    shuffle=False,
                                    num_workers=16,
                                    pin_memory=True,
                                    collate_fn=lambda x: x)
        
        # Accel distributed
        test_dataloader = accel.prepare(test_dataloader)

        # progress bar
        prog_bar = tqdm(test_dataloader, disable=not accel.is_local_main_process, total=len(test_dataloader))

        # eval start
        for inputs in prog_bar:

            # memory opt
            memory_optimization()

            # Generate
            with torch.inference_mode():

                # trol
                _inputs = model.eval_process(inputs=inputs,
                                             data=data,
                                             tokenizer=tokenizer,
                                             device=accel.device,
                                             img_token_number=args.img_token_num)
                generate_ids = model.generate(**_inputs, do_sample=False, num_beams=5, max_new_tokens=get_max_new_tokens(data), use_cache=True)

            # image visualization
            # imim = inputs[0]['image'].cpu().permute(1,2,0).numpy()
            decoded_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=False)
            
            # save predictions
            all_predictions = [output_filtering(x, model) for x in decoded_text]
            for x in inputs:
                if 'image' in x:
                    del x['image']
            evaluator.process(inputs, all_predictions)

        # wait for everyone
        print(f"[Device: {accel.device}] Finished!")
        accel.wait_for_everyone()

        # memory opt
        memory_optimization()

        # evaluate on dataset
        results[data] = evaluator.evaluate(args.link, data, accel)
    
    accel.print(results)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', 
                        default='mm-vet', 
                        help='all|vqav2|gqa|sqa|vizwiz|textvqa|pope|mme|mmbench|mmbench_cn|qbench|mm-vet|mmmu|mathvista|ai2d|hallusionbench|chartqa|seed|llava|mmstar|mathverse|visualwebbench',)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--link', default='TroL-1.8B', type=str)
    args = parser.parse_args()

    # resolution
    if not '3.8B' in args.link:
        args.resol = 490
        args.img_token_num = (args.resol // 14) ** 2
    else:
        args.resol = None
        args.img_token_num = None

    # test
    test(args)