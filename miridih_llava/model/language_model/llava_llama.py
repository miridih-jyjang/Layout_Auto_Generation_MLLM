#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from miridih_llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, PAD_TOKEN_INDEX


class LlavaConfig(LlamaConfig):
    model_type = "miridih_llava"

class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs
    
class LlavaLlamaForCausalLM_v5(LlavaLlamaForCausalLM):
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, past_key_values, labels, images, ele_images, img_mask):


        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels

        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            self.model.vision_tower.select_feature = 'patch'
            image_features = self.encode_images(images)
            del images
        
        if type(ele_images) is list or ele_images.ndim == 5: # B, sum(N), 3, 336, 336 where N=[N_1+N_2+..+N_B]
            ele_images = ele_images[img_mask.bool()] # [N_1+N_2+..+N_B] x 3, 336, 336
            if len(ele_images) > 0:
                self.model.vision_tower.select_feature = 'cls_patch'
                ele_image_features = self.encode_images(ele_images) # [N_1+N_2+..+N_B], 576, 4096
            
            # ele_image_features = torch.split(ele_image_features, split_sizes, dim=0) #B * (N, 576, 4096)
            # ele_image_features = torch.cat([ele_image_feature.unsqueeze(0) for ele_image_feature in ele_image_features], device=images.device, dim=0) # B, N, 576, 4096
            # img_count = img_mask.sum(1).cumsum(0) # torch.tensor([N_1, N_2+N_1, ..., sum(N)])
            # concat_ele_images, split_sizes = ele_images
            # ele_image_features = self.encode_images(concat_ele_images) # B*N, 576, 4096
            # ele_image_features = torch.split(ele_image_features, split_sizes, dim=0) #B * (N, 576, 4096)
            # ele_image_features = [tensor.to(images.device) for tensor in ele_image_features]
            
            # ele_image_features = self.encode_images(ele_images) # sum(N), 576, 4096
            img_count = img_mask.sum(1).cumsum(0)
            # num_imgs = torch.tensor([p.shape[0] for p in ele_images], device=input_ids.device)
            # ele_images = torch.cat(ele_images, dim=0).to(input_ids.device) # B*sum(N), 3, 336, 336
            # ele_image_features = self.encode_images(ele_images) # B*sum(N), 576, 4096
            # ele_image_features = torch.split(ele_image_features, num_imgs.tolist(), dim=0) #B * ([N_1, 576, 4096], [N_2, 576, 4096], ...)
            
        #     ele_image_features = [] # B *  [[N_1, 576, 4096], [N_2, 576, 4096], ...]
        #     for p in ele_images:
        #         ele_image_features.append(self.encode_images(p))
        else:
            # ele_image_features = self.encode_images(ele_images)
            ele_image_features = torch.empty(0).to(ele_images.device)
            
        new_input_embeds = []
        new_labels = [] if labels is not None else None
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_image_idx = 0
            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
                # multimodal LLM, but the current sample is not multimodal
                # FIXME: this is a hacky fix, for deepspeed zero3 to work
                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue
            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            original_num_image_token_indices = len(image_token_indices)
            # print("original_num_image_token_indices: ", original_num_image_token_indices)
            # assert(len(image_token_indices) == ele_image_features.shape[0]+1)
            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape
            while image_token_indices.numel() > 0:
                if cur_image_idx == 0:
                    cur_image_features = image_features[batch_idx]
                else:
                    try:
                        if batch_idx == 0:
                            cur_image_features = ele_image_features[:img_count[batch_idx]][cur_image_idx-1]
                        else:
                            cur_image_features = ele_image_features[img_count[batch_idx-1]:img_count[batch_idx]+1][cur_image_idx-1]
                    except:
                        print("img_count: {}\toriginal_num_image_token_indices: {}\nbatch_idx: {}\tcur_image_idx: {}\tele_image_features: {}".format(img_count, original_num_image_token_indices, batch_idx, cur_image_idx, ele_image_features.shape[0]))
                    
                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]
                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
                if labels is not None:
                    cur_new_labels.append(cur_labels)
            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)
        
        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        img_mask: Optional[torch.Tensor] = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, pixel_values, img_mask)

        del images, pixel_values

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = outputs[0] # [B, N_t, 4096] N_t: number of multi-modal tokens
        logits = self.lm_head(hidden_states)
        
        # # Zero-pad the logits to the specified length
        # pad_length = self.model.config.max_length
        # current_length = logits.size(1)
        # if current_length < pad_length:
        #     padding_size = pad_length - current_length
        #     logits = pad(logits, (0, 0, 0, padding_size), value=0)  # Pad on the sequence length dimension
        #     labels = pad(labels, (0, 0, 0, padding_size), value=IGNORE_INDEX)  # Pad on the sequence length dimension
        # else:
        #     logits = logits[:, :pad_length, :]
        #     labels = labels[:, :pad_length]
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "pixel_values": kwargs.get("pixel_values", None),
                "img_mask": kwargs.get("img_mask", None)
            }
        )
        return model_inputs

class LlavaLlamaForCausalLM_v6_4(LlavaLlamaForCausalLM_v5):

    def encode_features(self, image_features):
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, attention_mask, past_key_values, labels, images, ele_images, img_mask):
    #     # max_token_length = input_ids.shape[1]
    #     max_token_length = 4096
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
    #         return input_ids, attention_mask, past_key_values, None, labels
    #     # print("images: ", images.shape)
    #     if type(images) is list or images.ndim == 5:
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         image_features = [x.flatten(0, 1) for x in image_features]
    #     else:
    #         self.model.vision_tower.select_feature = 'patch'
    #         image_features = self.encode_images(images) # batch, pad_dim, feat_dim ex. 16, 576, 4096
    #         img_count = img_mask.sum(1).cumsum(0)
    #         del images
        
    #     if len(ele_images) > 0:
    #         ele_image_features = self.encode_features(ele_images) # [N_1+N_2+..+N_B], image_feature_dim, 4096 
    #     ele_image_features = ele_image_features[img_mask] # [N_1+N_2+..+N_B], image_feature_dim, 4096 
    #     new_input_embeds = []
    #     new_labels = [] if labels is not None else None
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         cur_image_idx = 0
    #         if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
    #             # multimodal LLM, but the current sample is not multimodal
    #             # FIXME: this is a hacky fix, for deepspeed zero3 to work
    #             half_len = cur_input_ids.shape[0] // 2
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
    #             cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             if labels is not None:
    #                 new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue
    #         image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    #         original_num_image_token_indices = len(image_token_indices)
            
    #         # cur_new_input_embeds = torch.full((image_features[batch_idx].shape[0]+cur_input_ids.shape[0]-1, image_features[batch_idx].shape[1]), PAD_TOKEN_INDEX, device=image_features.device, dtype=image_features.dtype)
    #         cur_new_input_embeds = []
    #         if labels is not None:
    #             cur_labels = labels[batch_idx]
    #             cur_new_labels = []
    #             assert cur_labels.shape == cur_input_ids.shape
    #         # print("cur_input_ids: {}\tbatch_idx: {}\timg_mask[batch_idx]: {}\tIMAGE_TOKEN_INDEX: {}".format(cur_input_ids.shape, batch_idx, img_mask[batch_idx].sum(), (cur_input_ids == IMAGE_TOKEN_INDEX).sum()))
            
    #         cur_input_ids[image_token_indices] = PAD_TOKEN_INDEX
            
    #         # print("cur_new_input_embeds: ", cur_new_input_embeds.shape)
            
    #         image_token_start = image_token_indices[0]
    #         if labels != None:
    #             labels[batch_idx][image_token_indices] = IGNORE_INDEX
    #             cur_new_labels.append(labels[batch_idx][:image_token_start])
    #             cur_new_labels.append(torch.full((image_features[batch_idx].shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #             cur_new_labels.append(labels[batch_idx][image_token_start+1:])
    #             cur_new_labels = torch.cat(cur_new_labels, dim=0)
    #             new_labels.append(cur_new_labels)
            
    #         # cur_new_input_embeds[:image_token_start] = self.get_model().embed_tokens(cur_input_ids[:image_token_start]).detach()
    #         # cur_new_input_embeds[image_token_start:image_token_start+image_features[batch_idx].shape[0]] = image_features[batch_idx].detach()
    #         # cur_new_input_embeds[image_token_start+image_features[batch_idx].shape[0]:] = self.get_model().embed_tokens(cur_input_ids[image_token_start+1:]).detach()
    #         cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]).detach())
    #         cur_new_input_embeds.append(image_features[batch_idx].squeeze(0))
    #         cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:]).detach())
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)

            
    #         shifted_ele_image_token_indices = (image_token_indices[1:] + (image_features[batch_idx].shape[0] - 1)).detach()
    #         if batch_idx == 0:
    #             cur_new_input_embeds[shifted_ele_image_token_indices] = ele_image_features[:len(image_token_indices[1:])].squeeze(1).to(cur_new_input_embeds.dtype).detach()
    #         else:
    #             cur_new_input_embeds[shifted_ele_image_token_indices] = ele_image_features[img_count[batch_idx-1]:img_count[batch_idx-1]+len(image_token_indices[1:])].squeeze(1).to(cur_new_input_embeds.dtype).detach()
                
    #         new_input_embeds.append(cur_new_input_embeds)
            
        
    #     new_input_embeds = torch.stack(new_input_embeds, dim=0)
    #     if labels != None:
    #         new_labels = torch.stack(new_labels, dim=0)
    #     else:
    #         new_labels = labels
        
    #     if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):

    #         if attention_mask is not None:
    #             new_attention_mask = []
    #             for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
    #                 new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
    #                 new_attention_mask.append(cur_new_attention_mask)
    #             attention_mask = torch.stack(new_attention_mask, dim=0)
    #             assert attention_mask.shape == new_labels.shape
    #     else:

    #         if attention_mask is not None:
    #             new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #             attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
    #             assert attention_mask.shape == new_input_embeds.shape[:2]

    #     new_input_embeds = new_input_embeds[:, :max_token_length, :]
    #     if labels != None:
    #         new_labels = new_labels[:, :max_token_length]
    #     attention_mask = attention_mask[:, :max_token_length]
        
    #     return None, attention_mask, past_key_values, new_input_embeds, new_labels
    
# AutoConfig.register("miridih_llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
