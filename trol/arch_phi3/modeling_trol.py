# System
import torch
from torch import nn
from utils.utils import *
import torch.utils.checkpoint
from typing import List, Optional, Tuple, Union
from transformers.modeling_utils import PreTrainedModel

# trol file
from .modeling_intern_vit import InternVisionModel
from .modeling_phi3 import Phi3ForCausalLM

# Dataclass & ModelOutput
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput


# Configuration
########################################################################################
import copy
from transformers.configuration_utils import PretrainedConfig
from .configuration_intern_vit import InternVisionConfig
from .configuration_phi3 import Phi3Config

class TroLConfig(PretrainedConfig):
    model_type = 'trol'
    is_composition = True

    def __init__(
            self,
            vision_config=None,
            llm_config=None,
            use_backbone_lora=0,
            use_llm_lora=0,
            pad2square=False,
            select_layer=-1,
            force_image_size=None,
            downsample_ratio=0.5,
            template=None,
            dynamic_image_size=False,
            use_thumbnail=False,
            ps_version='v1',
            min_dynamic_patch=1,
            max_dynamic_patch=6,
            **kwargs):
        super().__init__(**kwargs)
        self.vision_config = InternVisionConfig(**vision_config)
        self.llm_config = Phi3Config(**llm_config)
        self.use_backbone_lora = use_backbone_lora
        self.use_llm_lora = use_llm_lora
        self.pad2square = pad2square
        self.select_layer = select_layer
        self.force_image_size = force_image_size
        self.downsample_ratio = downsample_ratio
        self.template = template
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.ps_version = ps_version  # pixel shuffle version
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['llm_config'] = self.llm_config.to_dict()
        output['model_type'] = self.__class__.model_type
        output['use_backbone_lora'] = self.use_backbone_lora
        output['use_llm_lora'] = self.use_llm_lora
        output['pad2square'] = self.pad2square
        output['select_layer'] = self.select_layer
        output['force_image_size'] = self.force_image_size
        output['downsample_ratio'] = self.downsample_ratio
        output['template'] = self.template
        output['dynamic_image_size'] = self.dynamic_image_size
        output['use_thumbnail'] = self.use_thumbnail
        output['ps_version'] = self.ps_version
        output['min_dynamic_patch'] = self.min_dynamic_patch
        output['max_dynamic_patch'] = self.max_dynamic_patch
        return output
########################################################################################

@dataclass
class TroLCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None

class TroLForCausalLM(PreTrainedModel):
    config_class = TroLConfig

    def __init__(self, config):
        super().__init__(config)
        
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int((image_size // patch_size) ** 2 * (config.downsample_ratio ** 2))
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version

        self.vision_model = InternVisionModel(config.vision_config)
        self.language_model = Phi3ForCausalLM(config.llm_config)
        self.prompt_rule = {"system_start": "<s><|system|>\n",
                            "system_end": "<|end|>",
                            "user_start": "<|user|>\n",
                            "user_end": "<|end|>",
                            "assistant_start": "<|assistant|>\n",
                            "assistant_end": "<|end|>\n</s>",
                            "test_start": "<|assistant|>\n",
                            "test_end": "<|end|>",
                            "split": "\n",
                            }

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.vision_proj = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size)
        )
        
    def extract_feature(self, pixel_values):
        self.vision_model.eval()
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
        vit_embeds = vit_embeds[:, 1:, :]
        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        return vit_embeds

    def eval_process(
        self,
        inputs,
        data,
        tokenizer,
        device,
        img_token_number,
    ):
        batched_image = []
        batched_qa_prompt=[]
        for _input in inputs:

            # Visualization
            # imim = _input['image'].cpu().permute(1, 2, 0)

            # adding <image> to question if not included despite being an image, and adding system prompt and <tor> prompt 
            if 'image' in _input.keys() and not '<image>' in _input['question']: _input['question'] = '<image>\n' + _input['question']

            # making image prompt
            if 'image' in _input.keys() and _input['image'] != None:
                process_image = dynamic_preprocess(_input['image'].to(device))
                dynamic_process_image = torch.stack([dynamic_transform(image) for image in process_image]).to(device)
                img_token_number = dynamic_process_image.shape[0] * 256
                batched_image.append(dynamic_process_image)

            # make question and answer
            question = make_instruction(_input['question'], data, self.prompt_rule)

            # adding image special tokens to question
            if 'image' in _input.keys(): question = question.replace('<image>', '<img><IMG_CONTEXT></img>')

            # add bundle image tokens if it has <IMG_CONTEXT> token
            question = add_bundle_tokens(question, '<IMG_CONTEXT>', img_token_number) 

            batched_qa_prompt.append(question)

        '''For Final Outputs'''
        qa_prompts = tokenizer(batched_qa_prompt, padding='longest', return_tensors="pt", add_special_tokens=False)

        # [1] input_ids
        input_ids = qa_prompts.input_ids.to(device)
  
        # [2] attention_mask
        attention_mask = qa_prompts.attention_mask.to(device)

        if len(batched_image):
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    "image_features": self.extract_feature(torch.cat(batched_image, dim=0).to(device))
                    }
        else:
            return {"input_ids": input_ids, 
                    "attention_mask": attention_mask, 
                    }

    def _merge_input_embeds_with_image_features(self, image_features, inputs_embeds, input_ids):
        B, N, C = inputs_embeds.shape
        input_ids = input_ids.reshape(B * N)
        inputs_embeds = inputs_embeds.reshape(B * N, C)
        selected = torch.where(input_ids == self.config.image_token_index)
        assert selected[0].sum() != 0
        inputs_embeds[selected] = image_features.reshape(-1, C).to(inputs_embeds.device)
        inputs_embeds = inputs_embeds.reshape(B, N, C)
        return inputs_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        image_features: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TroLCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            try:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids).requires_grad_(False)
            except:
                inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if image_features is not None and input_ids.shape[1] != 1:

                image_features = self.vision_proj(image_features.to(inputs_embeds.dtype))
                inputs_embeds = self._merge_input_embeds_with_image_features(image_features, inputs_embeds, input_ids)
                
            # In case input_ids.shape[1] == 1 & image_features==None & past_key_values != None, we are in the case of
            # generation with cache
            elif past_key_values is not None and image_features is not None and input_ids.shape[1] == 1:
                # Retrieve the first layer to inspect the logits and mask out the hidden states
                # that are set to 0
                first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                # Get the target length
                target_length = input_ids.shape[1]
                past_length = first_layer_past_key_value.shape[-1]

                extended_attention_mask = torch.ones(
                    (attention_mask.shape[0], past_length),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

                # Filter out only the tokens that can be un-attended, this can happen
                # if one uses Llava + Fused modules where the cache on the
                # first iteration is already big enough, or if one passes custom cache
                valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                new_batch_index = batch_index[valid_indices]
                new_non_attended_tokens = non_attended_tokens[valid_indices]

                # Zero-out the places where we don't need to attend
                extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

                attention_mask = torch.cat((extended_attention_mask, attention_mask[:, -target_length:]), dim=1)
                position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return TroLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @torch.no_grad()
    def generate(
            self,
            image_features: Optional[torch.FloatTensor] = None,
            input_ids: Optional[torch.FloatTensor] = None,
            attention_mask: Optional[torch.LongTensor] = None,
            **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.config.image_token_index is not None
        if image_features is not None:
            vit_embeds = self.vision_proj(image_features)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.config.image_token_index)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            eos_token_id=self.config.eos_token_id,
            **generate_kwargs,
        )

        return outputs