import argparse
import torch
import json
import os
import re
from tqdm import tqdm
from setproctitle import setproctitle
# Need to call this before importing transformers.
import sys
sys.path.append("/workspace/Layout_Auto_Generation_MLLM")  
# from miridih_llava.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()
from miridih_llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from miridih_llava.conversation import conv_templates, SeparatorStyle
from miridih_llava.model.builder import load_pretrained_model
from miridih_llava.utils import disable_torch_init
from miridih_llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image, ImageDraw
from torch.nn.functional import pad
import requests
from io import BytesIO
from transformers import TextStreamer

bbox_extract =     re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)'")
bbox_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]+)\],\s*'layer':\s*([\d.]+),\s*'file_name':\s*'([^']*)'")
# bbox_src_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
bbox_src_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]+)\],\s*'layer':\s*(\d+(?:\.\d+)?),\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
bbox_src_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
bbox_layer_IMG_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'\[IMG(\d+)\]':\s*'<image>',\s*'file_name':\s*'([^']*)'")
bbox_extract_wo_filename = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'src':\s*'([^']*)'")
bbox_extract_wo_src_filename = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+)")
resolution_pattern = r"\[([0-9]+),([0-9]+)\]"

CLS2COLOR = {
    # "Miridih": {
    #     "generalsvg": "red", "text": "green", "shapesvg": "orange", "photo": "blue", "frameitem": "yellow",
    #     "lineshapeitem": "purple", "grid": "pink", "chart": "brown", "gif": "gray", "qrcode": "cyan", "video": "white",
    #     "barcode": "black", "youtube": "red", "basicsvg": "brown"
    # },
    "Miridih": {
        "svg": "red", "image": "green", "content": "orange", "code": "blue", "chart": "yellow",
        "lsi": "purple", "animation": "pink"
    },
    "QB": {
        "title": "red", "subtitle": "green", "item logo": "orange", "item": "blue", "decoration": "yellow",
        "text": "purple", "object": "brown", "frame": "gray", "false": "black"
    },
    "CGL": {
        "text": "red", "underlay": "green", "embellishment": "blue", "false": "black"
    },
    "Ad Banners": {
        "header": "red", "preheader": "green", "postheader": "blue", "body text": "orange", "disclaimer / footnote": "purple",
        "button": "pink", "callout": "brown", "logo": "gray", "false": "black"
    },
    "Crello": {
        "svgElement": "red", "textElement": "green", "imageElement": "blue", "coloredBackground": "black", "maskElement": "purple"
    }
}

def replace_keys(input_string, key_value_dict):
    for key, value in key_value_dict.items():
        # Create a regex pattern that matches the key, ignoring case
        pattern = re.compile(re.escape(key), re.IGNORECASE)
        # Replace the key with the value in the input string
        input_string = pattern.sub(value, input_string)
    return input_string

def get_json_response(response):
    for i in range(len(response)):
        if i < len(response) - 1 and response[i:i+2] == "[{":
            lo = i
        elif i > 1 and response[i-1:i+1] == "}]":
            hi = i
    try:
        string = response[lo:hi+1].replace("'", '"')
        json_response = json.loads(string)
    except:
        json_response = None
    return json_response

def stringTojson_v6(s):

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
    if 'src' in s:
        rects = bbox_extract_wo_filename.findall(s)
    else:
        rects = bbox_extract_wo_src_filename.findall(s)
    output = []
    for rect in rects:
        if len(rect) == 4:
            output.append({'label': rect[0], 'box': [clean_float_string(r) for r in rect[1].split(',')], 'layer': rect[2], 'src': rect[3]})
        else:
            output.append({'label': rect[0], 'box': [clean_float_string(r) for r in rect[1].split(',')], 'layer': rect[2]})
    
    return output

def draw_box(img, elems, cls2color, data_path):
    W, H = img.size
    rendering_image = img.copy().convert("RGBA")
    merge_image = Image.new("RGBA", rendering_image.size)
    # drawn_fill = img.copy()
    # draw_ol = ImageDraw.ImageDraw(drawn_outline)
    # draw_f = ImageDraw.ImageDraw(drawn_fill)
    for file_name, clss, box, layer in elems:
        
        template_id, page_num, ele_num = file_name.split('.')[0].split('_')
        page_num = int(page_num)
        image_file = f"{data_path}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{file_name}"
            
        overlay_img = Image.open(image_file).convert("RGBA")
        # color = cls2color[clss.lower()]
        try:
            left, top, right, bottom = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        except:
            continue
        ele_width, ele_height = W*(right-left), H*(bottom-top)
        overlay_img = overlay_img.resize((max(1, round(ele_width)), max(1,round(ele_height))))
        _, _, _, overlay_img_mask = overlay_img.split()
        rendering_image.paste(overlay_img, (int(left*W), int(top*H)), overlay_img_mask)
        merge_image = Image.alpha_composite(merge_image, rendering_image)
        # _box = int(left * W), int(top * H), int(right * W), int(bottom * H)
        # draw_ol.rectangle(_box, fill=None, outline=color)
        # draw_f.rectangle(_box, fill=color)
    # drawn_outline = drawn_outline.convert("RGBA")
    # drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    return merge_image
def sort_key(item):
    # Split the path and file name, then extract the number after the last underscore and before .png
    layer = item[3]  # Get file name (e.g., '5953615d95a7a863ddce1423_4.png')
    file_name = item[0]
    number = int(layer)  # Extract '4' as integer
    return (number, file_name)


def draw_boxmap(json_response, valid_eles, invalid_eles, background_image, cls2color, data_path):
    ele_num = min(len(json_response), len(valid_eles))
    inval_cls_box = [(invalid_ele['file_name'], invalid_ele['label'], [invalid_ele['x1'], invalid_ele['y1'], invalid_ele['x2'], invalid_ele['y2']], invalid_ele['layer']) for invalid_ele in invalid_eles]
    cls_box = [(valid_ele['file_name'], elem['label'], elem['box'], elem['layer']) for valid_ele, elem in zip(valid_eles[:ele_num], json_response[:ele_num])]
    cls_box = cls_box + inval_cls_box
    cls_box = sorted(cls_box, key=sort_key)

    # print(cls_box)
    drawn = draw_box(background_image, cls_box, cls2color, data_path)
    return drawn.convert("RGB")

# Code from the second script

def load_image(image_file):
    if image_file.startswith('http://') or image_file.startswith('https://'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def online_rendering(image_folder, skin_img_file, annotations, i_entry, args, task):
    rendering_image = Image.open(os.path.join(image_folder, skin_img_file))
    # W, H = rendering_image.size
    global W, H
    rendering_image = rendering_image.resize((W, H))
    merge_image = Image.new("RGBA", rendering_image.size)
    img_width, img_height = rendering_image.size
    for idx, anno in enumerate(annotations):
        ele_img_file = anno['file_name']
        template_id, page_num, ele_num = ele_img_file.split('.')[0].split('_')
        page_num = int(page_num)
        image_file = f"{image_folder}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{ele_img_file}"
        
        overlay_img = Image.open(image_file).convert("RGBA")
        ele_width, ele_height = W*(float(anno['x2'])-float(anno['x1'])), H*(float(anno['y2']) - float(anno['y1']))
            
        x_offset, y_offset = W*float(anno['x1']), H*float(anno['y1'])
        overlay_img = overlay_img.resize((max(1, round(ele_width)), max(1,round(ele_height))))
        _, _, _, overlay_img_mask = overlay_img.split()
        rendering_image.paste(overlay_img, (int(x_offset), int(y_offset)), overlay_img_mask)
        merge_image = Image.alpha_composite(merge_image, rendering_image)
        
    if args.debug and len(annotations) > 0 and i_entry < 10:    
        image_file = f"data/miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_01.png"
        
        if (task == 'refine') or (task == 'complete'): # completion & refinement
            merge_image.convert('RGB').save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_{task}_input.jpg"))
        # if not os.path.isfile(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_thumbnail.jpg")): # conditional
        #     thumbnail_img = Image.open(image_file).convert('RGB')
        #     thumbnail_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_thumbnail.jpg"))
    return merge_image

def extract_elements(bbox_html):
    # Extract all bounding boxes using the provided pattern
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

        return invalid_elements, valid_elements
    
def extract_unmasked_elements(bbox_html):
        matches = bbox_layer_IMG_extract.findall(bbox_html)
        file_attributes = []
        for match in matches:
            box = match[1].split(',')
            if "''" in box:
                continue
            file_attributes.append(
                    {
                        "file_name": match[4],
                        "label": match[0],
                        "x1": box[0],
                        "y1": box[1],
                        "x2": box[2],
                        "y2": box[3],
                        "layer": match[2],
                    }  
                )

        return file_attributes
    


def remove_elements(conversations, filenames):
    def remove_filenames(text):
        for filename_dict in filenames:
            if "Sure!" in text or "JSON file" in text:
                label = filename_dict['label']
                x1 = filename_dict['x1']
                y1 = filename_dict['y1']
                x2 = filename_dict['x2']
                y2 = filename_dict['y2']
                file_name = filename_dict['file_name']

                if 'src' in filename_dict:
                    src = filename_dict['src']
                    
                    if '<image>' in text:
                        if 'layer' in filename_dict:
                            layer = filename_dict['layer']
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.\d+)\],\s*'layer':\s*(" + re.escape(layer) + 
                                r"|\d+),\s*\[IMG\d+\]:\s*'<image>',\s*'file_name':\s*'" + re.escape(file_name) +
                                r"',\s*'src':\s*'" + re.escape(src) + r"'\}")
                        else:
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.\d+),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.\d+)\],\s*\[IMG\d+\]:\s*'<image>',\s*'file_name':\s*'" + re.escape(file_name) +
                                r"',\s*'src':\s*'" + re.escape(src) + r"'\}")
                            
                    else:
                        if 'layer' in filename_dict:
                            layer = filename_dict['layer']
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.?\d{0,4})\s*\],\s*'layer':\s*(" +
                                re.escape(layer) + r"|\d+),\s*'file_name':\s*'" + re.escape(file_name) +
                                r"',\s*'src':\s*'" + re.escape(src) + r"'\}\n")
                        else:
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.?\d{0,4})\s*\],\s*'file_name':\s*'" + re.escape(file_name) +
                                r"',\s*'src':\s*'" + re.escape(src) + r"'\}\n")
                            
                
                else:
                    if "'<image>'," in text:
                        if 'layer' in filename_dict:
                            layer = filename_dict['layer']
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(''|[-\d.]+),\s*(''|[-\d.]+),\s*(''|[-\d.]+),\s*(''|[-\d.]+)\],\s*'layer':\s*(" + 
                                re.escape(layer) + r"|\d*|''),\s*'\[IMG\d+\]':\s*'<image>',\s*'file_name':\s*'" + 
                                re.escape(file_name) + r"'\}")
                        else:
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(''|[-\d.]+),\s*(''|[-\d.]+),\s*(''|[-\d.]+),\s*(''|[-\d.]+)\],\s*'\[IMG\d+\]':\s*'<image>',\s*'file_name':\s*'" + 
                                re.escape(file_name) + r"'\}")
                            
                    else:
                        if 'layer' in filename_dict:
                            layer = filename_dict['layer']
                            # Create a regex pattern to match JSON objects with the same label and box coordinates
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.?\d{0,4})\s*\],\s*'layer':\s*(" + re.escape(layer) + 
                                r"|\d*|''),\s*'file_name':\s*'" + re.escape(file_name) + r"'\}")
                        else:
                            removal_pattern = (r"\{'label':\s*'" + re.escape(label) +
                                r"',\s*'box':\s*\[\s*(" + re.escape(format(x1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y1, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(x2, '.4f')) + r"|''|\d*\.?\d{0,4}),\s*(" +
                                re.escape(format(y2, '.4f')) + r"|''|\d*\.?\d{0,4})\s*\],\s*'file_name':\s*'" + re.escape(file_name) + r"'\}")

            else:
                img_ref = filename_dict['src']       # This is the image reference (e.g., "[IMG1]", "[IMG2]")
                pattern = r"\[IMG(\d+)\]"
                image_number = re.search(pattern, img_ref).group(1) # This is the integer part (e.g., "1" or "2")
                
                # Create the exact pattern for removal based on the match   
                removal_pattern = "image " + image_number + " is " + re.escape(img_ref) + " <image>.\n?"       

            # Substitute the pattern with an empty string
            text = re.sub(removal_pattern, '', text)
    
        # Clean up any extra newlines or spaces that might result from the removal
        text = re.sub(r'\n\s*\n', '\n', text).strip()
            
        return text
    
    for conversation in conversations:
        if 'value' in conversation:
            conversation['value'] = remove_filenames(conversation['value'])
    return conversations

def merge_lists_without_overlap(list1, list2):
    # Use a dictionary to track unique dictionaries by file_name
    merged_dict = {d['file_name']: d for d in list1}
    
    # Update the dictionary with entries from the second list
    merged_dict.update({d['file_name']: d for d in list2})
    
    # Convert the dictionary back to a list of dictionaries
    merged_list = list(merged_dict.values())
    
    return merged_list

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

def main(args):
    setproctitle('jooyoung-jang')
    # Model
    disable_torch_init()

    data = json.load(open(args.json_file, 'r', encoding='utf-8'))
    
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)
    with open(args.ele_cache_path, 'r') as f:
        ele_cache = json.load(f)
    
    if 'llama-2' in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    if args.conv_mode is not None and conv_mode != args.conv_mode:
        print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
    else:
        args.conv_mode = conv_mode

    conv = conv_templates[args.conv_mode].copy()
    if "mpt" in model_name.lower():
        roles = ('user', 'assistant')
    else:
        roles = conv.roles
        
    ret, gt = {}, {}
    
    os.makedirs(os.path.join('/'.join(args.output_file.split('/')[:-1]), 'gt'), exist_ok=True)
    os.makedirs(os.path.join('/'.join(args.output_file.split('/')[:-1]), 'input_complete'), exist_ok=True)
    os.makedirs(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{args.output_file.split('/')[-1].replace('.json', '')}"), exist_ok=True)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    erase_file_name = r",\s*'file_name':\s*'[^']*'"
    

    for i_entry, entry in enumerate(tqdm(data, desc="Processing entries")):
        
        if entry['id'] not in ret:
            ret[entry['id']] = []
            gt[entry['id']] = []

        conv = conv_templates[args.conv_mode].copy()
        
        invalid_filenames, valid_filenames = extract_elements(entry['conversations'][1]['value'])
        pixel_values = []
        for element in valid_filenames:
            ele_file = os.path.basename(element['file_name'])
            pixel_values.append(ele_cache[ele_file]) 
        try:
            img_mask = torch.ones(len(pixel_values)).bool().to(model.device)
        except:
            print("Empty element images in ", entry['id'])
            continue        
        merged_filelist = merge_lists_without_overlap(extract_unmasked_elements(entry['conversations'][0]['value']), invalid_filenames)

        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        template_id, page_num = entry['id'].split('_')
        template_id = int(template_id)
        page_num = int(page_num)
        
        # Added part #
        resolution_match = re.findall(resolution_pattern, entry['conversations'][0]['value'])
        global W, H
        W, H = tuple(map(int, resolution_match[0]))
        
        ##############

        if ('refine' in entry['image']) or ('complete' in entry['image']):
            image_file = f"miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
            image = online_rendering(args.data_path, image_file, merged_filelist, i_entry, args, entry['image']).convert('RGB')
        else:
            if os.path.isfile(f"{args.data_path}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                image_file = f"miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
            else:
                image_file = f"miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"

            if len(invalid_filenames) > 0:
                image = online_rendering(args.data_path, image_file, invalid_filenames, i_entry, args, entry['image']).convert('RGB')
            else:
                image = Image.open(os.path.join(args.data_path, image_file)).convert('RGB')

        image = image.resize((W, H))
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)


        # preprocessing prompt
        ## 1. remove invalid elements
        entry['conversations']= remove_elements(entry['conversations'], invalid_filenames)
        
        ## 2. remove file_name
        entry['conversations'][0]['value'] = re.sub(erase_file_name, '', entry['conversations'][0]['value'])
        entry['conversations'][1]['value'] = re.sub(erase_file_name, '', entry['conversations'][1]['value'])   
        
        ## 3. change valid elements
        entry['conversations'][0]['value'] = re.sub(r'(\d+)\sforeground elements', f'{len(valid_filenames)} foreground elements', entry['conversations'][0]['value'])

        # inp = '\n'.join(entry["conversations"][0]['value'].split('\n')[1:])
        inp = entry["conversations"][0]['value']
        conv.append_message(conv.roles[0], inp)
        # if image is not None:
        #     # first message
        #     if model.config.mm_use_im_start_end:
        #         inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        #     else:
        #         inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        #     conv.append_message(conv.roles[0], inp)
        #     # image = None
        # else:
        #     # later messages
        #     conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        
        if "posterllava/posterllava_v0" in args.model_path:
            prompt = replace_keys(prompt, MIRIDIH2QB)

        
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

        ele_img_tensors = torch.tensor(pixel_values).to(image_tensor.device)
        ele_img_tensors = ele_img_tensors.to(torch.float16)  # Converts to float16 (half precision)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                pixel_values=ele_img_tensors.unsqueeze(0),
                img_mask=img_mask.unsqueeze(0),
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        ret[entry['id']].append(stringTojson_v6(outputs))
        gt[entry['id']].append(stringTojson_v6(entry['conversations'][1]['value']))
        if args.debug or args.image_out:
            # for r in ret[entry['id']][-1]:
                # for v in valid_filenames:
                #     if v['src'] == r['src']:
                #         r['file_name'] = v['file_name']
                #         break

            # if not os.path.isfile(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"input_complete/{template_id}_{page_num:01}.jpg")):
            #     try:
            #         # thumbnail_image_file = os.path.join(args.data_path, image_file.replace('_1.png', '_0.png'))
            #         # thumbnail_img = Image.open(thumbnail_image_file).convert('RGB')
            #         # thumbnail_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"gt/{template_id}_{page_num:01}.jpg"))
            #         image.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"input_complete/{template_id}_{page_num:01}.jpg"))
            #     except:
            #         print("{} is not existing!!".format(f"{template_id:08}_{page_num:01}.jpg"))
            if entry['image'] == 'refine': # refinement
                if len(invalid_filenames) > 0:
                    args.debug = False
                    image = online_rendering(args.data_path, image_file, invalid_filenames, i_entry, args, entry['image']).convert('RGB')
                    args.debug = True
                else:
                    image = Image.open(os.path.join(args.data_path, image_file)).convert('RGB')
                    image = image.resize((W, H))
            if "posterllava/posterllava_v0" in args.model_path:
                drawn_img = draw_boxmap(ret[entry['id']][-1], valid_filenames, image, CLS2COLOR["QB"], args.data_path)  # Adjust the category as needed
            else:
                inv_filenames = [f['file_name'] for f in invalid_filenames]
                drawn_img = draw_boxmap(ret[entry['id']][-1], valid_filenames, invalid_filenames, image, CLS2COLOR["Miridih"], args.data_path)  # Adjust the category as needed
            drawn_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{args.output_file.split('/')[-1].replace('.json', '')}/{template_id}_{page_num:01}.jpg"))
                
            
                


    with open(args.output_file, 'w', encoding='utf-8') as fout:
        json.dump(ret, fout, ensure_ascii=False, indent=2)
    
    with open(args.output_file.replace('.json', '_gt.json'), 'w', encoding='utf-8') as fout:
        json.dump(gt, fout, ensure_ascii=False, indent=2)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--image-aspect-ratio", type=str, default='pad')
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--json-file", type=str, required=True)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--ele_cache_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-out", action="store_true")
    args = parser.parse_args()
    print("args: ", args)
    setproctitle('jooyoung-jang')
    main(args)
