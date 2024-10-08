# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
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
import numpy as np
import sys
sys.path.append("/workspace/Layout_Auto_Generation_MLLM")  
from setproctitle import setproctitle
from scipy.optimize import linear_sum_assignment
from torch import BoolTensor, FloatTensor
from miridih_llava.model.fid import LayoutNet
from safetensors.torch import load_model
from pytorch_fid.fid_score import calculate_frechet_distance
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Tuple, Union
import torch
import wandb
import transformers
# import datasets
from miridih_llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from miridih_llava.train.llava_trainer import LLaVATrainer
from miridih_llava import conversation as conversation_lib
from miridih_llava.model import *
from miridih_llava.mm_utils import tokenizer_image_token
from PIL import Image
from deepspeed.runtime.utils import see_memory_usage
from torch.nn.functional import pad
# datasets.config.IN_MEMORY_MAX_SIZE = 300 *1024 *1024 *1024

local_rank = None
def compute_metrics(eval_pred, device):
    # pred, gt, mask, category, WH, device
    geo_pred, cat_pred, mask_pred = eval_pred.predictions
    geo_gts, cat_gts, mask_gts = eval_pred.label_ids
    
    evaluator = Evaluator(device)
    result_dict = evaluator(geo_pred, geo_gts, cat_pred, cat_gts, mask_pred)
    
    return result_dict
                                      
def rank0_print(*args):
    if local_rank == 0:
        print(*args)
Layout = Tuple[np.ndarray, np.ndarray]

def convert_xywh_to_ltrb(bbox: Union[np.ndarray, FloatTensor]):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]

def compute_iou(
    box_1: Union[np.ndarray, FloatTensor],
    box_2: Union[np.ndarray, FloatTensor],
    generalized: bool = False,
) -> Union[np.ndarray, FloatTensor]:
    # box_1: [N, 4]  box_2: [N, 4]

    if isinstance(box_1, np.ndarray):
        lib = np
    elif isinstance(box_1, FloatTensor):
        lib = torch
    else:
        raise NotImplementedError(type(box_1))

    l1, t1, r1, b1 = convert_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = convert_xywh_to_ltrb(box_2.T)
    a1, a2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)

    # intersection
    l_max = lib.maximum(l1, l2)
    r_min = lib.minimum(r1, r2)
    t_max = lib.maximum(t1, t2)
    b_min = lib.minimum(b1, b2)
    cond = (l_max < r_min) & (t_max < b_min)
    ai = lib.where(cond, (r_min - l_max) * (b_min - t_max), lib.zeros_like(a1[0]))

    au = a1 + a2 - ai
    iou = ai / au

    if not generalized:
        return iou

    # outer region
    l_min = lib.minimum(l1, l2)
    r_max = lib.maximum(r1, r2)
    t_min = lib.minimum(t1, t2)
    b_max = lib.maximum(b1, b2)
    ac = (r_max - l_min) * (b_max - t_min)

    giou = iou - (ac - au) / ac

    return giou

def __compute_mean_iou_for_layout(layout_1: Layout, layout_2: Layout) -> float:
    score = 0.0
    count = 0  # To keep track of the number of IoU comparisons
    (bi, li), (bj, lj) = layout_1, layout_2
    
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        n, m = len(_bi), len(_bj)
        # Skip this iteration if either _bi or _bj is empty
        if len(_bi) == 0 or len(_bj) == 0:
            continue
        
        if n == 0:
            continue
        
        ii, jj = np.meshgrid(range(n), range(m))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, m)
        
        # Set NaN values to a small number
        iou[np.isnan(iou)] = 1e-6
        
        # Use the linear sum assignment to maximize IoU
        ii, jj = linear_sum_assignment(iou, maximize=True)
        
        # Sum the IoU for the matched pairs
        score += iou[ii, jj].sum().item()
        
        # Increment the count by the number of matched pairs
        count += len(ii)
    
    # Return the mean IoU by dividing the total score by the total number of comparisons
    if count == 0:
        return 0.0  # To handle edge cases where there are no comparisons
    
    return score / count

def __compute_maximum_iou_for_layout(layout_1: Layout, layout_2: Layout) -> float:
    score = 0.0
    (bi, li), (bj, lj) = layout_1, layout_2
    N = len(bi)
    for l in list(set(li.tolist())):
        _bi = bi[np.where(li == l)]
        _bj = bj[np.where(lj == l)]
        
        # Skip this iteration if either _bi or _bj is empty
        if len(_bi) == 0 or len(_bj) == 0:
            continue
        
        n, m = len(_bi), len(_bj)
        ii, jj = np.meshgrid(range(n), range(m))
        ii, jj = ii.flatten(), jj.flatten()
        iou = compute_iou(_bi[ii], _bj[jj]).reshape(n, m)
        # note: maximize is supported only when scipy >= 1.4
        iou[np.isnan(iou)] = 1e-6
        ii, jj = linear_sum_assignment(iou, maximize=True)
        score += iou[ii, jj].sum().item()
    if N == 0:
        return 0.0
    return score / N

def preprocess_layouts(layouts, types):
    processed_layouts = []
    for layout, type_ in zip(layouts, types):
        # PyTorch 텐서를 CPU로 이동시키고 NumPy 배열로 변환합니다.
        layout_cpu = layout.cpu().numpy() if layout.is_cuda else layout.numpy()
        type_cpu = type_.cpu().numpy() if type_.is_cuda else type_.numpy()
        
        # type이 0이 아닌 요소의 인덱스를 찾습니다.
        valid_indices = np.where(type_cpu != 0)[0]
        # type이 0이 아닌 요소에 해당하는 valid의 요소만을 선택합니다.
        valid_layout = layout_cpu[valid_indices]
        valid_type = type_cpu[valid_indices]
        processed_layouts.append((valid_layout, valid_type))
    return processed_layouts

def mean_iou_one_by_one(layout_set_1, layout_set_2, types_1, types_2):
    
    layout_set_1 = preprocess_layouts(layout_set_1, types_1)
    layout_set_2 = preprocess_layouts(layout_set_2, types_2)
    
    total_iou = 0.0
    for layout_1, layout_2 in zip(layout_set_1, layout_set_2):
        max_iou = __compute_mean_iou_for_layout(layout_1, layout_2)
        total_iou += max_iou
    average_iou = total_iou / len(layout_set_1)
    return average_iou

def maximum_iou_one_by_one(layout_set_1, layout_set_2, types_1, types_2):
    
    layout_set_1 = preprocess_layouts(layout_set_1, types_1)
    layout_set_2 = preprocess_layouts(layout_set_2, types_2)
    
    total_iou = 0.0
    for layout_1, layout_2 in zip(layout_set_1, layout_set_2):
        max_iou = __compute_maximum_iou_for_layout(layout_1, layout_2)
        total_iou += max_iou
    average_iou = total_iou / len(layout_set_1)
    return average_iou

class Evaluator:
    def __init__(self, device):
        super().__init__()
        fid_model = LayoutNet(num_label=7).eval().requires_grad_(False).to(device)
        load_model(fid_model, 'pretrained/model.safetensors', strict=True)
        self.fid_model = fid_model

    def compute_fid_score(self, real_feats, fake_feats):

        real_mu = np.mean(real_feats, axis=0)
        real_sigma = np.cov(real_feats, rowvar=False)

        fake_mu = np.mean(fake_feats, axis=0)
        fake_sigma = np.cov(fake_feats, rowvar=False)

        return calculate_frechet_distance(real_mu, real_sigma, fake_mu, fake_sigma)

    def denormalize_boundary(self, normalized_geometry):
        norm_x1, norm_y1, norm_x2, norm_y2 = normalized_geometry[:, 0], normalized_geometry[:, 1], normalized_geometry[:, 2], normalized_geometry[:, 3]
        
        # Renormalize coordinates from [0, 1] to [-1, 1]
        norm_x1 = 2 * norm_x1 - 1
        norm_y1 = 2 * norm_y1 - 1
        norm_x2 = 2 * norm_x2 - 1
        norm_y2 = 2 * norm_y2 - 1
        
        denormalized_geo = torch.stack([norm_x1, norm_y1, norm_x2, norm_y2], dim=-1)
        
        return denormalized_geo
    
    def get_fid_score(self, pred_geometry, gt_geometry, pred_category, gt_category, mask):
        gt_feats, pd_feats = [], []

        for stack_pd_geometry, stack_gt_geometry, stack_pred_category, stack_gt_category, stack_mask in zip(pred_geometry, gt_geometry, pred_category, gt_category, mask):
            for pd_geo, gt_geo, pred_cat, gt_cat, mask_ in zip(stack_pd_geometry[:,:,:-1], stack_gt_geometry[:,:,:-1], stack_pred_category, stack_gt_category, stack_mask):
                
                pd_geo = self.denormalize_boundary(pd_geo)
                gt_geo = self.denormalize_boundary(gt_geo)
                pd_geo = pd_geo.float()
                gt_geo = gt_geo.float()
                pred_cat = pred_cat.float()
                gt_cat = gt_cat.float()
                mask_ = mask_.bool()

                selected_gt_geometry = torch.masked_select(gt_geo, mask_[:, :-1])
                selected_pd_geometry = torch.masked_select(pd_geo, mask_[:, :-1])
                selected_gt_cat = torch.masked_select(gt_cat, mask_[:, -1])
                selected_pd_cat = torch.masked_select(pred_cat, mask_[:, -1])

                input_gt_geometry = selected_gt_geometry.reshape(1, -1, 4)
                input_pd_geometry = selected_pd_geometry.reshape(1, -1, 4)
                input_gt_class = selected_gt_cat.unsqueeze(0).long()  # Convert to long tensor
                input_pd_class = selected_pd_cat.unsqueeze(0).long()  # Convert to long tensor
                padding_mask = mask_[:, -1]
                padding_mask = padding_mask[[padding_mask!=False]]

                with torch.no_grad():
                    true_feats = self.fid_model.inference(input_gt_geometry, input_gt_class, ~padding_mask)
                    false_feats = self.fid_model.inference(input_pd_geometry, input_pd_class, ~padding_mask)

                    gt_feats.append(true_feats)
                    pd_feats.append(false_feats)
        
        gt_features = torch.stack(gt_feats).cpu().squeeze().numpy()
        pd_features = torch.stack(pd_feats).cpu().squeeze().numpy()
        fid_score = self.compute_fid_score(gt_features, pd_features)

        return fid_score

    def get_layer_accuracy(self, pred_layer, gt_layer, mask):

        total_accuracy = 0.
        total_cnt = 0.

        for stack_pd_layer, stack_gt_layer, stack_mask in zip(pred_layer, gt_layer, mask):
            
            for pred_z, gt_z, mask_ in zip(stack_pd_layer[:,:, -1], stack_gt_layer[:,:, -1], stack_mask[:,:, -1]):
                selected_gt_z = torch.masked_select(gt_z, mask_)
                selected_pd_z = torch.masked_select(pred_z, mask_)
                # accuracy = torch.sum(selected_gt_z.argsort() == selected_pd_z.argsort()) / len(selected_pd_z)
                if len(selected_pd_z) > 0:  # Check if the selected tensor is non-empty
                    accuracy = torch.sum(selected_gt_z.argsort() == selected_pd_z.argsort()) / len(selected_pd_z)
                    total_accuracy += accuracy
                    total_cnt += 1
                else:
                    if len(selected_gt_z) > 0:
                        total_cnt += 1


                # total_accuracy += accuracy
                # total_cnt += 1

        return total_accuracy / total_cnt
    
    def denormalize(self, normalized_geometry):
        norm_x1, norm_y1, norm_x2, norm_y2 = normalized_geometry[:, :, 0], normalized_geometry[:, :, 1], normalized_geometry[:, :, 2], normalized_geometry[:, :, 3]
        
        # Renormalize coordinates from [0, 1] to [-1, 1]
        
        # Calculate width, height, and center coordinates
        w = norm_x2 - norm_x1
        h = norm_y2 - norm_y1
        x = norm_x1 + w/2
        y = norm_y1 + h/2
        xywh = torch.cat([x,y,w,h], dim=0)
        
        return xywh.transpose(0,1).unsqueeze(0)

    def get_ious(self, pred_geometry, gt_geometry, pred_category, gt_category, mask):
        total_max_iou = 0.
        total_mean_iou = 0.
        total_cnt = 0.
        '''
        normalized_x = geometry_array[:, 0] + normalized_w / 2
        normalized_y = geometry_array[:, 1] + normalized_h / 2
        normalized_w = geometry_array[:, 2] - geometry_array[:, 0]
        normalized_h = geometry_array[:, 3] - geometry_array[:, 1]
        '''
        for stack_pd_geometry, stack_gt_geometry, stack_pred_category, stack_gt_category, stack_mask in zip(pred_geometry, gt_geometry, pred_category, gt_category, mask):
            for pd_geo, gt_geo, pred_cat, gt_cat, mask_ in zip(stack_pd_geometry[:,:,:-1], stack_gt_geometry[:,:,:-1], stack_pred_category, stack_gt_category, stack_mask[:,:,:-1]):
     
                selected_gt_geometry = torch.masked_select(gt_geo, mask_)
                selected_pd_geometry = torch.masked_select(pd_geo, mask_)
                selected_gt_cat = torch.masked_select(gt_cat, mask_[:, -1])
                selected_pd_cat = torch.masked_select(pred_cat, mask_[:, -1])

                input_gt_geometry = selected_gt_geometry.reshape(1, -1, 4)
                input_pd_geometry = selected_pd_geometry.reshape(1, -1, 4)
                input_gt_class = selected_gt_cat.unsqueeze(0)
                input_pd_class = selected_pd_cat.unsqueeze(0)

                denorm_gt_geometry = self.denormalize(input_gt_geometry)
                denorm_pd_geometry = self.denormalize(input_pd_geometry)

                max_iou = maximum_iou_one_by_one(denorm_gt_geometry.detach(), denorm_pd_geometry.detach(), input_gt_class.detach(), input_pd_class.detach())
                mean_iou = mean_iou_one_by_one(denorm_gt_geometry.detach(), denorm_pd_geometry.detach(), input_gt_class.detach(), input_pd_class.detach())
                
                total_max_iou += max_iou
                total_mean_iou += mean_iou
                total_cnt += 1

        return total_max_iou / total_cnt, total_mean_iou / total_cnt

    def __call__(self, geo_pred, geo_gts, cat_pred, cat_gts, mask_pred):
          
        # geometry = (x,y,w,h,z)
        # pred_geometry, gt_geometry, cat, mask, WH
        layer_accuracy = self.get_layer_accuracy(geo_pred, geo_gts, mask_pred)
        print("layer_accuracy: ", layer_accuracy)
        max_iou, mean_iou = self.get_ious(geo_pred, geo_gts, cat_pred, cat_gts, mask_pred)
        print("max_iou: {}\tmean_iou: {}".format(max_iou, mean_iou))
        fid_score = self.get_fid_score(geo_pred, geo_gts, cat_pred, cat_gts, mask_pred)
        print("fid_score: ", fid_score)
        return {
            "layer_accuracy": layer_accuracy,
            "fid_score": fid_score,
            "max_iou": max_iou,
            "mean_iou": mean_iou
        }


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    dev_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    data_version: str = field(default='v1',
                              metadata={"help": "Version of Training data"})
    ele_cache_path: str = field(default=None,
                                metadata={"help": "Element clip encoded vectors."})
    eval_ele_cache_path: str = field(default=None,
                                metadata={"help": "Element clip encoded vectors."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    torch_empty_cache_steps: int = field(default=None)
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    batch_eval_metrics: bool = field(
        default=False,
        metadata={"help": "Break eval metrics calculation into batches to save memory."},
    )
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    exp_name: str = field(default="", metadata={"help": "Experiment name"})

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        
        del image # jyj
        
        return data_dict

def pad_images(image, max_length):
    mask = torch.zeros(max_length).bool()

    # Handle the case where the image is a 0-dimensional tensor (scalar)
    if image.numel() == 0:
        # Pad the empty image to a size [1, 3, 336, 336] or similar default size
        image = torch.zeros(1, 3, 336, 336)  # Replace with the appropriate shape
    else:
        # Update mask
        mask[:image.shape[0]] = True
    
    pad_len = max_length - image.shape[0]

    # If the image is empty or scalar, skip padding and return
    if image.shape[0] == 0:
        return image, mask
    
    
    # Apply padding
    image = pad(image, (0, 0, 0, 0, 0, 0, 0, pad_len))  # padding along the first dimension

    return image, mask


@dataclass
class DataCollatorForSupervisedDataset_v6_4(object):
    
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def pad_sequences(self, sequences, max_len, padding_value=0):
        """
        Pads a list of sequences to a fixed max length.
        
        Args:
        sequences (list of torch.Tensor): List of sequences (1D tensors) of various lengths.
        max_len (int): The fixed length to pad each sequence to.
        padding_value (float or int, optional): The value used for padding. Default is 0.
        
        Returns:
        torch.Tensor: A tensor of shape (batch_size, max_len) with padded sequences.
        """
        # Initialize a tensor with padding_value, shape = (batch_size, max_len)
        batch_size = len(sequences)
        padded_sequences = torch.full((batch_size, max_len), padding_value)
        
        for i, seq in enumerate(sequences):
            seq_len = seq.size(0)  # Length of the current sequence
            if seq_len <= max_len:
                # Copy the sequence into the padded tensor (truncate if necessary)
                padded_sequences[i, :seq_len] = seq
            else:
                # Optionally, handle cases where sequences are longer than max_len
                padded_sequences[i, :] = seq[:max_len]

        return padded_sequences

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, pixel_values = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "pixel_values"))
        # input_ids = self.pad_sequences(
        #     input_ids,
        #     max_len=self.tokenizer.model_max_length,
        #     padding_value=self.tokenizer.pad_token_id)
        # mask_list, image_list = [], []
        # # max_image_length = max([img.shape[0] for img in pixel_values])
        # max_image_length = MAX_ELE_NUM_CRELLO
        # if type(pixel_values) == list: 
        #     # mask_list: [batch_size, MAX_ELE_NUM_CRELLO, 1]
        #     mask_list = [torch.tensor([True] * len(pv) + [False] * (MAX_ELE_NUM_CRELLO - len(pv))).bool() for pv in pixel_values]
        #     # pixel_values: [batch_size, num_elements, [1x1024]] -> [batch_size, MAX_ELE_NUM_CRELLO, [1x1024]]
        #     image_list = [torch.tensor(pv + [[[0] * len(pv[0][0])]] * (MAX_ELE_NUM_CRELLO - len(pv))) if len(pv) < MAX_ELE_NUM_CRELLO else torch.tensor(pv) for pv in pixel_values]

        # else:
        #     for img in pixel_values:
        #         image, img_mask = pad_images(img,max_image_length)
        #         image_list.append(image)
        #         mask_list.append(img_mask)
        # pixel_values = torch.stack(image_list).to(input_ids.device)
        # img_mask = torch.stack(mask_list).to(input_ids.device)
       
        # labels = self.pad_sequences(labels,
        #                             max_len=self.tokenizer.model_max_length,
                                    # padding_value=IGNORE_INDEX)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        mask_list, image_list = [], []
        max_image_length = max([len(pv) for pv in pixel_values])
        # max_image_length = MAX_ELE_NUM_CRELLO
        if type(pixel_values) == list: 
            # mask_list: [batch_size, MAX_ELE_NUM_CRELLO, 1]
            mask_list = [torch.tensor([True] * len(pv) + [False] * (max_image_length - len(pv))).bool() for pv in pixel_values]
            # pixel_values: [batch_size, num_elements, [1x1024]] -> [batch_size, MAX_ELE_NUM_CRELLO, [1x1024]]
            image_list = [torch.tensor(pv + [[[0] * len(pv[0][0])]] * (max_image_length - len(pv))) if len(pv) < max_image_length else torch.tensor(pv) for pv in pixel_values]

        else:
            for img in pixel_values:
                image, img_mask = pad_images(img,max_image_length)
                image_list.append(image)
                mask_list.append(img_mask).to(input_ids.device)
        img_mask = torch.stack(mask_list).to(input_ids.device)
        pixel_values = torch.stack(image_list).to(input_ids.device)

        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
            img_mask = img_mask
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    
@dataclass
class DataCollatorForSupervisedDataset_v6(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, pixel_values = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "pixel_values"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        mask_list, image_list = [], []
        max_image_length = max([img.shape[0] for img in pixel_values])
        for img in pixel_values:
            image, img_mask = pad_images(img,max_image_length)
            image_list.append(image)
            mask_list.append(img_mask)
        pixel_values = torch.stack(image_list).to(input_ids.device)
        img_mask = torch.stack(mask_list).to(input_ids.device)
        # pixel_values = torch.nn.utils.rnn.pad_sequence(
        #     pixel_values,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # num_imgs = torch.tensor([len(p) for p in pixel_values]).to(input_ids.device)
        # pixel_values = torch.cat(pixel_values, dim=0).to(input_ids.device)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
            # num_imgs=num_imgs
            img_mask = img_mask
            # pixel_values=[pixel_values, num_imgs]
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch
    
@dataclass
class DataCollatorForSupervisedDataset_v5(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, pixel_values = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "pixel_values"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        mask_list, image_list = [], []
        max_image_length = max([img.shape[0] for img in pixel_values])
        for img in pixel_values:
            image, img_mask = pad_images(img,max_image_length)
            image_list.append(image)
            mask_list.append(img_mask)
        pixel_values = torch.stack(image_list).to(input_ids.device)
        img_mask = torch.stack(mask_list).to(input_ids.device)
        # pixel_values = torch.nn.utils.rnn.pad_sequence(
        #     pixel_values,
        #     batch_first=True,
        #     padding_value=self.tokenizer.pad_token_id)
        # num_imgs = torch.tensor([len(p) for p in pixel_values]).to(input_ids.device)
        # pixel_values = torch.cat(pixel_values, dim=0).to(input_ids.device)
        
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            pixel_values=pixel_values,
            # num_imgs=num_imgs
            img_mask = img_mask
            # pixel_values=[pixel_values, num_imgs]
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if 'v6.4' in data_args.data_version or 'v6.5' in data_args.data_version:
        if 'miridih' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v6_4 import LazyRealTimeRenderingDataset
        elif 'crello' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v6_4_crello import LazyRealTimeRenderingDataset
    elif 'v6' in data_args.data_version:
        if 'miridih' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v6 import LazyRealTimeRenderingDataset
        elif 'crello' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v6_crello import LazyRealTimeRenderingDataset
    elif 'v5' in data_args.data_version:
        if 'miridih' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v5 import LazyRealTimeRenderingDataset
        elif 'crello' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v5_crello import LazyRealTimeRenderingDataset
    elif data_args.data_version == 'v4':
        if 'miridih' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v4 import LazyRealTimeRenderingDataset
        elif 'crello' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v4_crello import LazyRealTimeRenderingDataset
    elif data_args.data_version == 'v3':
        if 'miridih' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender import LazyRealTimeRenderingDataset
        elif 'crello' in data_args.data_path:
            from miridih_llava.data.lazyRealtimeRender_v3_crello import LazyRealTimeRenderingDataset
    else:
        print("error for version")
    if 'v6.4' in data_args.data_version or 'v6.5' in data_args.data_version:
        train_dataset = LazyRealTimeRenderingDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    ele_cache_path=data_args.ele_cache_path,
                                    data_args=data_args)
        dev_dataset = LazyRealTimeRenderingDataset(tokenizer=tokenizer,
                                    data_path=data_args.dev_data_path,
                                    ele_cache_path=data_args.eval_ele_cache_path,
                                    data_args=data_args)
    else:
        train_dataset = LazyRealTimeRenderingDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
        dev_dataset = LazyRealTimeRenderingDataset(tokenizer=tokenizer,
                                    data_path=data_args.dev_data_path,
                                    data_args=data_args)
        
    if 'v6.4' in data_args.data_version or 'v6.5' in data_args.data_version:
        data_collator = DataCollatorForSupervisedDataset_v6_4(tokenizer=tokenizer)
    elif 'v6' in data_args.data_version:
        data_collator = DataCollatorForSupervisedDataset_v6(tokenizer=tokenizer)
    elif 'v5' in data_args.data_version:
        data_collator = DataCollatorForSupervisedDataset_v5(tokenizer=tokenizer)
    else:
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=dev_dataset,
                data_collator=data_collator)


def eval():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.exp_name == "":
        wandb.init(project='posterLlava-crello-instruction')
    else:
        print("experiment: ", training_args.exp_name)
        wandb.init(project='posterLlava-crello-instruction', name=training_args.exp_name)
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    see_memory_usage('Memory usage before model creation', True)
    
    bnb_model_from_pretrained_args = {}
    checkpoint_dir = training_args.output_dir
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMPTForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            if "v6.4" in data_args.data_version or 'v6.5' in data_args.data_version:
                model = LlavaLlamaForCausalLM_v6_4.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            elif "v6" in data_args.data_version:
                model = LlavaLlamaForCausalLM_v5.from_pretrained(
                checkpoint_dir,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            elif "v5" in data_args.data_version:
                model = LlavaLlamaForCausalLM_v5.from_pretrained(
                checkpoint_dir,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
            else:
                model = LlavaLlamaForCausalLM.from_pretrained(
                    checkpoint_dir,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    see_memory_usage('Memory usage after model creation', True)

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    see_memory_usage('Memory usage after tokenizer creation', True)

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    see_memory_usage('Memory usage after initializing vision model', True)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    see_memory_usage('Memory usage after data module creation', True)
    
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    content_aware_layout_generation_compute_metrics=compute_metrics,
                    **data_module)
    
    see_memory_usage('Memory usage after LLaVA-trainer creation', True)

    content_aware_layout_generation_metrics = trainer.content_aware_layout_generation_evaluate()
    reduced_eval_log = {
            "eval":{
                "Z Accuracy": content_aware_layout_generation_metrics['eval_layer_accuracy'],
                "FID Score": content_aware_layout_generation_metrics['eval_fid_score'],
                "Max IOU": content_aware_layout_generation_metrics['eval_max_iou'],
                "Mean IOU": content_aware_layout_generation_metrics['eval_mean_iou']
            }
        }

    wandb.log(reduced_eval_log)
    print(f"layer_accuracy: {content_aware_layout_generation_metrics['eval_layer_accuracy']:.4f} | fid_score: {content_aware_layout_generation_metrics['eval_fid_score']:.4f} | max_iou: {content_aware_layout_generation_metrics['eval_max_iou']:.4f} | mean_iou: {content_aware_layout_generation_metrics['eval_mean_iou']:.4f}")

    trainer.save_state()

    model.config.use_cache = True
    wandb.finish()

def eval_after_prediction(dataset_name, pred_json, gt_json):
    from miridih_llava.train.trainer_pt_utils import EvalLoopContainer
    from miridih_llava.train.trainer_utils import EvalLoopOutput
    from miridih_llava.constants import IGNORE_INDEX, MAX_ELE_NUM_CRELLO
    from helper.global_var import CONVERTED_DATASET_META

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_geo_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
    all_cat_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
    all_mask_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
    all_geo_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
    all_cat_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
    all_mask_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        
    all_preds = json.load(open(pred_json, 'r'))
    all_gts = json.load(open(gt_json, 'r'))
    geo_preds, cat_preds, mask_preds = [], [], []
    geo_gts, cat_gts = [], []
    for key_pd, key_gt in zip(all_preds, all_gts): # sample
        assert key_pd == key_gt
        sample_preds, sample_gts = all_preds[key_pd], all_gts[key_gt]
        
        for preds, gts in zip(sample_preds, sample_gts): # template
            geo_pred, cat_pred = [[-1, -1, -1, -1, -1]] * MAX_ELE_NUM_CRELLO, [-1] * MAX_ELE_NUM_CRELLO
            geo_gt, cat_gt = [[-1, -1, -1, -1, -1]] * MAX_ELE_NUM_CRELLO, [-1] * MAX_ELE_NUM_CRELLO
            mask_pred = [[False, False, False, False, False]] * MAX_ELE_NUM_CRELLO
            # mask_pred[:len(preds)] = [True, True, True, True, True]
            for idx, (pred, gt) in enumerate(zip(preds, gts)): # element
                geo_pred[idx], cat_pred[idx] = [float(b) for b in pred['box']] + [int(pred['layer'])], CONVERTED_DATASET_META[dataset_name][pred['label']]
                geo_gt[idx], cat_gt[idx] = [float(b) for b in gt['box']] + [int(pred['layer'])], CONVERTED_DATASET_META[dataset_name][gt['label']]
                mask_pred[idx]= [True, True, True, True, True]
        geo_preds.append(torch.tensor(geo_pred).to(device).unsqueeze(0))
        cat_preds.append(torch.tensor(cat_pred).to(device).unsqueeze(0))
        geo_gts.append(torch.tensor(geo_gt).to(device).unsqueeze(0))
        cat_gts.append(torch.tensor(cat_gt).to(device).unsqueeze(0))     
        mask_preds.append(torch.tensor(mask_pred).to(device).unsqueeze(0))
        
       
    logits = (geo_preds, cat_preds, mask_preds)
    labels = (geo_gts, cat_gts, mask_preds)

    num_samples = len(geo_preds)
    return EvalLoopOutput(predictions=logits, label_ids=labels, metrics=None, num_samples=num_samples)
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="miridih")
    parser.add_argument("--pred_json", type=str, default=None)
    parser.add_argument("--gt_json", type=str, default=None)
    args = parser.parse_args()
    setproctitle('jooyoung-jang')
    eval()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # output = eval_after_prediction(args.dataset_name, args.pred_json, args.gt_json)
    # compute_metrics(output, device)
    
