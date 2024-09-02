import os, torch
import re, sys, json
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
sys.path.append("/workspace/Layout_Auto_Generation_MLLM")  
from miridih_llava.model.multimodal_encoder.clip_encoder import CLIPVisionTower
from transformers import LlamaConfig
import pandas as pd

class LlavaConfig(LlamaConfig):
    model_type = "miridih_llava",
    _name_or_path= "/data/checkpoints/jjy/llava_v1.5_7b_crello_v4_1e/"
    architectures= ["LlavaLlamaForCausalLM"]
    bos_token_id= 1
    eos_token_id= 2
    freeze_mm_mlp_adapter= False
    freeze_mm_vision_resampler= False
    hidden_act= "silu"
    hidden_size= 4096
    image_aspect_ratio= "pad"
    image_grid_pinpoints= None
    initializer_range= 0.02
    intermediate_size= 11008
    max_length= 4096
    max_position_embeddings= 4096
    mm_hidden_size= 1024
    mm_projector_lr= None
    mm_projector_type= "mlp2x_gelu"
    mm_resampler_type= None
    mm_use_im_patch_token= False
    mm_use_im_start_end= False
    mm_vision_select_feature= "patch"
    mm_vision_select_layer= -2
    mm_vision_tower= "openai/clip-vit-large-patch14-336"
    model_type= "miridih_llava"
    num_attention_heads= 32
    num_hidden_layers= 32
    num_key_value_heads= 32
    pad_token_id= 0
    pretraining_tp= 1
    rms_norm_eps= 1e-05
    rope_scaling= None
    tie_word_embeddings= False
    torch_dtype= "bfloat16"
    transformers_version= "4.31.0"
    tune_mm_mlp_adapter= False
    tune_mm_vision_resampler= False
    unfreeze_mm_vision_tower= False
    use_cache= True
    use_mm_proj= True
    vocab_size= 32000
            
# Regular expression pattern to extract the number of elements
pattern = r'place\s(\d+)\sforeground elements'
  

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
    
def load_and_preprocess_image(image_path, image_processor, resolution=None):
    # image_path:  -> 'data/crello-v4/images/5ef1b92f499b85dcc7bb736e_2.png'
    try:
        new_images = []
        image = Image.open(image_path).convert("RGB")
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        new_images.append(image)
        if all(x.shape == new_images[0].shape for x in new_images):
            new_images = torch.stack(new_images, dim=0)
        return new_images
    except (FileNotFoundError, Image.DecompressionBombWarning, UnidentifiedImageError):
        return None
        
def feature_select(image_forward_outs):
    image_features = image_forward_outs.hidden_states[-2]
    image_features = image_features[:, 1:]
    return image_features

def main(use_resized_img=False):
    csv_path = "data/crello-v4/raw/train/train.csv"
    base_path = "data/crello-v4"

    csv_data = pd.read_csv(csv_path)
    csv_data.loc[:, 'priority'] = csv_data['priority'].astype(float).astype(int)
    csv_data = csv_data.sort_values(by='priority')

    batch_size = 64
    output_json = "./train_element_clip_features.json"
    vision_tower_name = 'openai/clip-vit-large-patch14-336'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name)
    kwarg = {'delay_load': False}
    vision_tower = CLIPVisionTower(vision_tower_name, LlavaConfig, **kwarg)
    # image_processor = CLIPImageProcessor.from_pretrained(vision_tower_name)
    image_processor = vision_tower.image_processor
    vision_tower.requires_grad_(False)
    vision_tower.to(device=device, dtype=torch.float16)

    results = {}
    # iterate image files of elements and extract element image features
    for start_idx in tqdm(range(0, len(csv_data), batch_size), desc="Loading images in batches"):
        end_idx = start_idx + batch_size
        batch_data = csv_data.iloc[start_idx:end_idx]
        
        image_tensors = []
        image_paths = []

        for _, image_info in batch_data.iterrows():        
            template_id = image_info['reformat_image_file_name'].split('/')[-1].split('_')[0]
            if template_id in results:
                continue
         
            image_path = os.path.join(base_path, image_info['reformat_image_file_name'])
            image_tensor = load_and_preprocess_image(image_path, image_processor)
            image_tensor = image_tensor.to(vision_tower.device, dtype=torch.float16)
            if image_tensor is not None:
                image_tensors.append(image_tensor)
                image_paths.append(image_path)
        
        batch_tensors = torch.cat(image_tensors).to(device)
        with torch.no_grad():
            # vision transformer 사용
            batch_features = vision_tower(batch_tensors.to(dtype=batch_tensors.dtype))

        for k, image_features in enumerate(batch_features):
            image_features = image_features.cpu().numpy().tolist()
            image_path = image_paths[k]
            if template_id not in results:
                results[template_id] = {}
            results[template_id][os.path.basename(image_path)] = image_features
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":

    use_resized_img = False
    main(use_resized_img)
