import os, re
import torch
import torch.nn.functional as F
import time
from torch.utils.data import Sampler
from torch import nn
from typing import Optional, Dict, Any, Union  # Import typing module
from torch.utils.data import DataLoader, Dataset
from transformers import EvalPrediction
from miridih_llava.train.trainer_pt_utils import EvalLoopContainer
from transformers.trainer import denumpify_detensorize, find_batch_size, IterableDatasetShard
from miridih_llava.train.trainer_utils import EvalLoopOutput
from transformers import Trainer
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ShardedDDPOption,
    ALL_LAYERNORM_LAYERS,
    logger,
)
import wandb
from miridih_llava.constants import IGNORE_INDEX, MAX_ELE_NUM_CRELLO
from helper.global_var import CONVERTED_DATASET_META
from typing import List, Optional
import gc
from miridih_llava.mm_utils import KeywordsStoppingCriteria
from transformers.deepspeed import deepspeed_init
import deepspeed
import transformers
from miridih_llava.train.llama_flash_attn_monkey_patch import original_forward, original_prepare_mask, replace_llama_attn_with_flash_attn
from tqdm import tqdm

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

bbox_extract =     re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)'")
bbox_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'file_name':\s*'([^']*)'")
bbox_src_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
bbox_src_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")


class LLaVATrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.content_aware_layout_generation_compute_metrics = kwargs['content_aware_layout_generation_compute_metrics']
        if 'crello' in kwargs['args'].exp_name:
            self.dataset_name = 'crello'
        elif 'miridih' in kwargs['args'].exp_name:
            self.dataset_name = 'miridih'
        else:
            print("[WARNINGS] should rename expermient name for proper evaluation!!")
            
        del kwargs['content_aware_layout_generation_compute_metrics']
        self.bbox_extract_wo_filename = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'src':\s*'([^']*)'")
        super().__init__(*args, **kwargs)
    
    def _preprocess_logits_for_metrics(self, logits, labels):
        ## detokenizer the logits to predictions
        geo_preds, cat_preds, mask_preds, geo_gts, cat_gts, mask_gts = [], [], [], [], [], []
        # assert(logits)
        answer_start_idx = torch.nonzero(labels[0]!=IGNORE_INDEX)
        valid_logits = logits[answer_start_idx[0], answer_start_idx[1]+1:] # <\s>
        try:
            assert(valid_logits[0][0] == 1) # bos
        except:
            print("[ERROR] valid_logits[0][0]: ", valid_logits[0][0])
        # valid_logits = logits[:, labels[labels==IGNORE_INDEX].shape[0]:]
        try:
            eos_idx = torch.nonzero(valid_logits==self.tokenizer.eos_token_id)
        except:
            eos_idx = [0, len(valid_logits[0])-1]
        padded_logits = valid_logits[0][:eos_idx[1]+1]
        
        # if not stopped properly: eos token is more than two / eos token is not in the last position
        # if len(torch.nonzero(logits == self.tokenizer.eos_token_id)) > 1 or labels[labels==IGNORE_INDEX].shape[0] == 0:
        #     valid_logits = logits[:, labels[labels==IGNORE_INDEX].shape[0]:]
        #     first_idx = torch.nonzero((valid_logits==self.tokenizer.eos_token_id))[0]
        #     padded_logits = valid_logits[0][:first_idx[1]+1]
        # else:
        #     padded_logits = logits[:, labels[labels==IGNORE_INDEX].shape[0]:].squeeze(0)
            
        output = self.tokenizer.decode(padded_logits).strip()
        gt = self.tokenizer.decode(labels[:, answer_start_idx+1:eos_idx[1]+1].squeeze(0)).strip()
        print("output: ", output)
        print("gt: ", gt)
        def stringToTensor_v6(s):
            def clean_float_string(s):
                # Step 1: Remove leading/trailing whitespace
                s = s.strip()

                # Step 2: Replace multiple decimal points
                # Find the first decimal point and split the string around it
                if s.count('.') > 1:
                    # Keep only the first occurrence of a decimal point
                    parts = s.split('.', 1)
                    # Remove additional decimal points from the second part
                    s = parts[0] + '.' + parts[1].replace('.', '')

                return s
            # Find all rect elements and their attributes within the SVG body
            rects = self.bbox_extract_wo_filename.findall(s)
            geo, cat = [[-1, -1, -1, -1, -1]] * MAX_ELE_NUM_CRELLO, [-1] * MAX_ELE_NUM_CRELLO
            mask = [[False, False, False, False, False]] * MAX_ELE_NUM_CRELLO
            # torch.nn.utils.rnn.pad_sequence
            for idx, rect in enumerate(rects):
                try:
                    geo[idx] = [float(clean_float_string(r)) for r in rect[1].split(',')] + [int(rect[2])]
                except:
                    print("s: {}\trects: {}".format(s, rects))
                cat[idx] = CONVERTED_DATASET_META[self.dataset_name][rect[0]]
            mask[:len(rects)] = [[True, True, True, True, True]] * len(rects)
            
            geo = torch.tensor(geo).to(logits.device)
            cat = torch.tensor(cat).to(logits.device)
            mask = torch.tensor(mask).to(logits.device)
            
            return geo, cat, mask
        
        # for output, gt in zip(outputs, gts_string):
        geo_pred, cat_pred, mask_pred = stringToTensor_v6(output)
        geo_preds.append(geo_pred)
        cat_preds.append(cat_pred)
        mask_preds.append(mask_pred)
        geo_gt, cat_gt, mask_gt = stringToTensor_v6(gt)
        geo_gts.append(geo_gt)
        cat_gts.append(cat_gt)
        mask_gts.append(mask_gt)

        geo_preds = torch.stack(geo_preds)
        cat_preds = torch.stack(cat_preds)
        mask_preds = torch.stack(mask_preds)
        geo_gts = torch.stack(geo_gts)
        cat_gts = torch.stack(cat_gts)
        mask_gts = torch.stack(mask_gts)
        
        return geo_preds, cat_preds, mask_preds, geo_gts, cat_gts, mask_gts

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log and self.state.global_step > self._globalstep_last_logged:

            logs: Dict[str, float] = {}

            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        content_aware_layout_generation_metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)
            # metrics = self.evaluate(trial, ignore_keys_for_eval)
            # try:
            # content_aware_layout_generation_metrics = self._content_aware_layout_generation_evaluate(trial, ignore_keys_for_eval)
            # self.log(content_aware_layout_generation_metrics)
            # except:
            #     print("[WARNING] custom evaluation not working")
            
        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            # self._save_checkpoint(model, trial, metrics=content_aware_layout_generation_metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _content_aware_layout_generation_evaluate(self, trial, ignore_keys_for_eval, skip_scheduler=False):
        metrics = self.content_aware_layout_generation_evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and not skip_scheduler:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            try:
                self.lr_scheduler.step(metrics[metric_to_check])
            except KeyError as exc:
                raise KeyError(
                    f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                    f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                ) from exc
        return metrics

    def content_aware_layout_generation_evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else None
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.content_aware_layout_generation_evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        self._memory_tracker.start()
        
        eval_dataloader = self.get_content_aware_layout_generation_eval_dataloader(eval_dataset)

        start_time = time.time()
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.content_aware_layout_generation_evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Content Aware Layout Generation Evaluation",
            prediction_loss_only=True if self.content_aware_layout_generation_compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def inference_step(self, model, inputs, prediction_loss_only, eval_only, ignore_keys=None):
        # Move inputs to the correct device
        inputs = self._prepare_inputs(inputs)
        
        # For text generation, we usually don't calculate loss, so skip that part.
        # We'll also ignore `prediction_loss_only` for now as we're focusing on generation.
        
        # Call model.generate instead of a forward pass
        answer_start_idx = torch.nonzero(inputs['labels']!=IGNORE_INDEX)[0] # first index's 2nd dim which is seq_idx
        
        inputs['input_ids'] = inputs['input_ids'][:, :answer_start_idx[1]] 
        # inputs['input_ids'][0][-1] = self.tokenizer.eos_token_id
        # try:
        #     assert(inputs['input_ids'][0][-1] == 2) # 2 = eos
        # except:
        #     print("[ERROR] inputs['input_ids'][0][-1]: ", inputs['input_ids'][0][-1])
            
        # print("input: ", self.tokenizer.decode(inputs['input_ids'][0]).strip())

        if eval_only:
            with torch.inference_mode():
                stop_str = '</s>'
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs['input_ids'])
                generated_tokens = model.generate(
                    input_ids=inputs['input_ids'],
                    images=inputs['images'],
                    pixel_values=inputs['pixel_values'],
                    img_mask=inputs['img_mask'],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=4096,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )
        else:
            with torch.no_grad():
                stop_str = '</s>'
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, inputs['input_ids'])
                generated_tokens = model.generate(
                    input_ids=inputs['input_ids'],
                    images=inputs['images'],
                    pixel_values=inputs['pixel_values'],
                    img_mask=inputs['img_mask'],
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=4096,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria]
                )

        return None, generated_tokens, inputs['labels']
    
    def content_aware_layout_generation_evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        args = self.args
        
        if self.deepspeed == None:
            eval_only = True
        else:
            eval_only = False
            
        # ori_ds_enabled, ori_ds = self.is_deepspeed_enabled, self.deepspeed
        # self.is_deepspeed_enabled, self.deepspeed = False, None

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)
        # init deepspeed inference engine
        # model = deepspeed.init_inference(
        #     model=self.model,      # Transformers models
        #     mp_size=1,        # Number of GPU
        #     dtype=torch.float16, # dtype of the weights (fp16)
        #     # replace_method="auto", # Lets DS autmatically identify the layer to replace
        #     # replace_with_kernel_inject=True, # replace the model with the kernel injector
        # )
                
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            start_time = time.time()
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )
            self.model_preparation_time = round(time.time() - start_time, 4)

            if self.is_fsdp_enabled:
                self.model = model

            if model is not self.model:
                self.model_wrapped = model

            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped
        
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        # batch_size = self.args.eval_batch_size
        batch_size = 1

        logger.info(f"\n***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = 1")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        all_losses = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_geo_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_cat_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_mask_preds = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_geo_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_cat_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_mask_labels = EvalLoopContainer(do_nested_concat=False, padding_index=-100)
        all_inputs = EvalLoopContainer(do_nested_concat=False, padding_index=-100)

        metrics = None

        observed_num_examples = 0
        thumbnail_vector, elements_vector = {}, {}  # self.cache(args, dataloader, model, description)
        
        self.gather_function = self.accelerator.gather_for_metrics

        # Restore the original methods before prediction
        transformers.models.llama.modeling_llama.LlamaAttention.forward = original_forward
        transformers.models.llama.modeling_llama.LlamaModel._prepare_decoder_attention_mask = original_prepare_mask
        
        num_tasks = 6
        
        for step, inputs in enumerate(tqdm(dataloader, desc="Processing evaluation")):
            if (step+1) % num_tasks != 0 and not eval_only:
                continue
            
            if step > 30:
                break
            
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                if batch_size is None:
                    batch_size = observed_batch_size
                
            inputs = self._prepare_inputs(inputs)
            # inputs have 'input_ids', 'labels', 'attention_mask', 'pixel_values', 'img_mask', 'images'
            losses, logits, labels = self.inference_step(model, inputs, prediction_loss_only, eval_only, ignore_keys=ignore_keys)
            
            # pad_length = self.args.model_max_length - logits.size(1)
            # logits = F.pad(logits, (0, pad_length), value=self.tokenizer.pad_token_id)
            # labels = F.pad(labels, (0, pad_length), value=IGNORE_INDEX)


            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if losses is not None:
                losses = self.gather_function((losses.repeat(batch_size)))
                all_losses.add(losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_inputs.add(inputs_decode)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self._preprocess_logits_for_metrics is not None:
                    try:
                        geo_preds, cat_preds, mask_preds, geo_gts, cat_gts, mask_gts = self._preprocess_logits_for_metrics(logits, labels)
                    except:
                        print("exception at {} step since there were no valid eos token for logits".format(step))
                        continue
                geo_preds = self.gather_function((geo_preds))
                cat_preds = self.gather_function((cat_preds))
                mask_preds = self.gather_function((mask_preds))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_geo_preds.add(geo_preds)
                    all_cat_preds.add(cat_preds)
                    all_mask_preds.add(mask_preds)
                geo_gts = self.gather_function((geo_gts))
                cat_gts = self.gather_function((cat_gts))
                mask_gts = self.gather_function((mask_gts))
                if not self.args.batch_eval_metrics or description == "Prediction":
                    all_geo_labels.add(geo_gts)
                    all_cat_labels.add(cat_gts)
                    all_mask_labels.add(mask_gts)
            # self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            
            logits = (geo_preds, cat_preds, mask_preds)
            labels = (geo_gts, cat_gts, mask_gts)

            if self.args.batch_eval_metrics:
                if self.content_aware_layout_generation_compute_metrics is not None and logits is not None and labels is not None:
                    is_last_step = self.accelerator.gradient_state.end_of_dataloader
                    if args.include_inputs_for_metrics:
                        metrics = self.content_aware_layout_generation_compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels, inputs=inputs),
                            compute_result=is_last_step,
                        )
                    else:
                        metrics = self.content_aware_layout_generation_compute_metrics(
                            EvalPrediction(predictions=logits, label_ids=labels),
                            compute_result=is_last_step,
                        )

                del losses, logits, labels, inputs
                torch.cuda.empty_cache()

            elif args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_geo_preds.to_cpu_and_numpy()
                all_cat_preds.to_cpu_and_numpy()
                all_mask_preds.to_cpu_and_numpy()
                all_geo_labels.to_cpu_and_numpy()
                all_cat_labels.to_cpu_and_numpy()
                all_mask_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()
                del losses, logits, labels, inputs
                torch.cuda.empty_cache()
        
         # Restor back to flash attention during training
        replace_llama_attn_with_flash_attn()
        
        if args.past_index and hasattr(self, "_past"):
            delattr(self, "_past")

        all_geo_preds = [t.to(dataloader.device) for t in all_geo_preds.tensors]
        all_cat_preds = [t.to(dataloader.device) for t in all_cat_preds.tensors]
        all_mask_preds = [t.to(dataloader.device) for t in all_mask_preds.tensors]
        all_geo_labels = [t.to(dataloader.device) for t in all_geo_labels.tensors]
        all_cat_labels = [t.to(dataloader.device) for t in all_cat_labels.tensors] 
        all_mask_labels = [t.to(dataloader.device) for t in all_mask_labels.tensors]
        all_inputs = all_inputs.tensors

        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples
        if (
            self.content_aware_layout_generation_compute_metrics is not None
            and all_geo_preds is not None
            and all_geo_labels is not None
            and not self.args.batch_eval_metrics
        ):
            if args.include_inputs_for_metrics:
                metrics = self.content_aware_layout_generation_compute_metrics(
                    EvalPrediction(predictions=(all_geo_preds, all_cat_preds, all_mask_preds), label_ids=(all_geo_labels, all_cat_labels, all_mask_labels), inputs=all_inputs)
                )
            else:
                metrics = self.content_aware_layout_generation_compute_metrics(EvalPrediction(predictions=(all_geo_preds, all_cat_preds, all_mask_preds), label_ids=(all_geo_labels, all_cat_labels, all_mask_labels)), device=dataloader.device)
        elif metrics is None:
            metrics = {}
        metrics = denumpify_detensorize(metrics)
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        # for key in list(metrics.keys()):
        #     if not (key.startswith(f'{metric_key_prefix}_Recall') or key.startswith(f'{metric_key_prefix}_MRR')):
        #         del metrics[key]

        # revert back to detraining mode
        # self.is_deepspeed_enabled, self.deepspeed = ori_ds_enabled, ori_ds
        return EvalLoopOutput(predictions=(all_geo_preds, all_cat_preds, all_mask_preds), label_ids=(all_geo_labels, all_cat_labels, all_mask_labels), metrics=metrics, num_samples=num_samples)

    def extract_elements(self, bbox_html):
        if 'src' in bbox_html:
            if 'layer' in bbox_html:
                matches = bbox_src_layer_extract.findall(bbox_html)
            else:
                matches = bbox_src_extract.findall(bbox_html)
        else:
            if 'layer' in bbox_html:
                matches = bbox_layer_extract.findall(bbox_html)
            else:
                matches = bbox_extract.findall(bbox_html)
        
        # Find and return invalid elements
        invalid_elements, valid_elements = [], []
        for match in matches:
            if 'layer' in bbox_html:
                box = match[1].split(',')
                category = match[0]
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                layer = match[2]
                file_name = match[3]
                if len(match) == 5:
                    src = match[4]
            else:
                box = match[1].split(',')
                category = match[0]
                x1 = float(box[0])
                y1 = float(box[1])
                x2 = float(box[2])
                y2 = float(box[3])
                file_name = match[2]
                if len(match) == 4:
                    src = match[3]

            temp_dict = {
                "file_name": file_name,
                "label": category,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
            if 'layer' in bbox_html:
                temp_dict["layer"] = layer
            if 'src' in bbox_html:
                temp_dict["src"] = src
            valid_elements.append(temp_dict)

        return valid_elements
    def get_content_aware_layout_generation_eval_dataloader(self, eval_dataset: Optional[Union[str, Dataset]] = None) -> DataLoader:
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        # if (
        #     hasattr(self, "_eval_dataloaders")
        #     and dataloader_key in self._eval_dataloaders
        # ):
        #     return self.accelerator.prepare(self._eval_dataloaders[dataloader_key])

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )
        data_collator = self.data_collator

        dataloader_params = {
            "batch_size": 1,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(eval_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_eval_sampler(eval_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        eval_dataloader = DataLoader(eval_dataset, **dataloader_params)
        return self.accelerator.prepare(eval_dataloader)
        # return eval_dataloader
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        # print("**** [3] Training Step ****")
        model.train()
        inputs = self._prepare_inputs(inputs)
        # print("images: {}\timg_mask: {}\tpixel_values: {}\tattention_mask: {}\tlabels: {}\tinput_ids: {}".format(inputs['images'].shape, inputs['img_mask'].shape, inputs['pixel_values'].shape, inputs['attention_mask'].shape, inputs['labels'].shape, inputs['input_ids'].shape))

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        if self.args.torch_empty_cache_steps is not None:
            # Clear GPU & CPU cache
            # torch.cuda.empty_cache()
            gc.collect()
            del inputs

        return loss.detach() / self.args.gradient_accumulation_steps

    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        # print("**** [4] Compute Loss ****")
        # print("images: {}\timg_mask: {}\tpixel_values: {}\tattention_mask: {}\tlabels: {}\tinput_ids: {}".format(inputs['images'].shape, inputs['img_mask'].shape, inputs['pixel_values'].shape, inputs['attention_mask'].shape, inputs['labels'].shape, inputs['input_ids'].shape))
        
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss