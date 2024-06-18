import torch
from config import *
from PIL import Image
from utils.utils import *
import torch.nn.functional as F
from trol.load_trol import load_trol
from torchvision.transforms.functional import pil_to_tensor

# model selection
link = 'TroL-7B' # [Select One] 'TroL-1.8B' | 'TroL-3.8B' | 'TroL-7B'

# User prompt
prompt_type="with_image" # Select one option "text_only", "with_image"
img_path='figures/demo.png'
question="What is the troll doing? Provide the detail in the image and imagine what the event happens."

# loading model
model, tokenizer = load_trol(link=link)
    
# cpu -> gpu
for param in model.parameters():
    if not param.is_cuda:
        param.data = param.to('cuda:0')

# prompt type -> input prompt
image_token_number = None
if prompt_type == 'with_image':
    # Image Load
    image = pil_to_tensor(Image.open(img_path).convert("RGB"))
    if not "3.8B" in link:
        image_token_number = 1225
        image = F.interpolate(image.unsqueeze(0), size=(490, 490), mode='bicubic').squeeze(0)
    inputs = [{'image': image, 'question': question}]
elif prompt_type=='text_only':
    inputs = [{'question': question}]

# Generate
with torch.inference_mode():
    _inputs = model.eval_process(inputs=inputs,
                                 data='demo',
                                 tokenizer=tokenizer,
                                 device='cuda:0',
                                 img_token_number=image_token_number)
    generate_ids = model.generate(**_inputs, max_new_tokens=256, use_cache=True)
    response = output_filtering(tokenizer.batch_decode(generate_ids, skip_special_tokens=False)[0], model)
print(response)