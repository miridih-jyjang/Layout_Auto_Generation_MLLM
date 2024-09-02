import argparse
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

from miridih_llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from miridih_llava.conversation import conv_templates, SeparatorStyle
from miridih_llava.model.builder import load_pretrained_model
from miridih_llava.utils import disable_torch_init
from miridih_llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image, ImageDraw

import requests
from io import BytesIO
from transformers import TextStreamer

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

def stringTojson_v2(s):
    # Find the SVG content within the <body> tags
    svg_content = re.search(r'<svg\s+([^>]+)>(.*?)<\/svg>', s, re.DOTALL)
    
    if not svg_content:
        print("No SVG content found")
        return None
    
    # Extract the attributes of the <svg> tag
    svg_attributes = svg_content.group(1)
    svg_body = svg_content.group(2)
    
    width = float(re.search(r'width=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
    height = float(re.search(r'height=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
    
    # Find all rect elements and their attributes within the SVG body
    rects = re.findall(r'<rect\s+([^>]+)\/>', svg_body)
    
    output = []
    for rect in rects:
        # Extract the attributes of each rect
        attributes = re.findall(r'([\w\-]+)=["\']([^"\']+)["\']', rect)
        rect_dict = {attr[0]: attr[1] for attr in attributes}
        
        # Map to the original desired format, using extracted width and height for normalization
        mapped_dict = {
            "label": rect_dict.get('data-category', ''),
            "box": [
                float(rect_dict.get('x', 0)) / width,  # Normalizing with extracted width
                float(rect_dict.get('y', 0)) / height,  # Normalizing with extracted height
                (float(rect_dict.get('x', 0)) + float(rect_dict.get('width', 0))) / width,
                (float(rect_dict.get('y', 0)) + float(rect_dict.get('height', 0))) / height
            ]
        }
        output.append(mapped_dict)
    
    return output



def stringTojson(s):
    for i in range(len(s)):
        if i < len(s) - 1 and s[i:i+2] == "[{":
            lo = i
        elif i > 1 and s[i-1:i+1] == "}]":
            hi = i

    tries, max_tries = 0, 1
    output = ''
    while tries < max_tries:
        try:
            string = s[lo:hi+1].replace("'", '"')
            output = json.loads(string)
            break
        except json.JSONDecodeError as e:
            tries += 1
            print(f"Tried for {tries} times, error parsing JSON: {e}")
        except UnboundLocalError as e:
            tries += 1
            print(f"Tried for {tries} times, error parsing JSON: {e}")
    
    return output

def draw_box(img, elems, cls2color):
    W, H = img.size
    rendering_image = img.copy().convert("RGBA")
    merge_image = Image.new("RGBA", rendering_image.size)
    # drawn_fill = img.copy()
    # draw_ol = ImageDraw.ImageDraw(drawn_outline)
    # draw_f = ImageDraw.ImageDraw(drawn_fill)
    for file_name, clss, box in elems:
        template_id, page_num = file_name.split('/')[-1].split('.')[0].split('_')
        page_num = int(page_num)
        image_file = f"data/crello-v3/images/{template_id}_1.png"
        overlay_img = Image.open(image_file).convert("RGBA")
        # color = cls2color[clss.lower()]
        left, top, right, bottom = box
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

def draw_boxmap(json_response, valid_filenames, background_image, cls2color):
    ele_num = min(len(json_response), len(valid_filenames))
    cls_box = [(file_name, elem['label'], elem['box']) for file_name, elem in zip(valid_filenames[:ele_num], json_response[:ele_num])]
    # print(cls_box)
    drawn = draw_box(background_image, cls_box, cls2color)
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
    merge_image = Image.new("RGBA", rendering_image.size)
    img_width, img_height = rendering_image.size
    for idx, anno in enumerate(annotations):
        ele_img_file = anno['file_name']
        template_id = ele_img_file.split('/')[-1].split('_')[0]
        page_num = 0
        image_file = f"{image_folder}/crello-v3/{ele_img_file}"
        overlay_img = Image.open(image_file).convert("RGBA")
        ele_width, ele_height = float(anno['width']), float(anno['height'])
        
        x_offset, y_offset = float(anno['x']), float(anno['y'])
        overlay_img = overlay_img.resize((max(1, round(ele_width)), max(1,round(ele_height))))
        _, _, _, overlay_img_mask = overlay_img.split()
        rendering_image.paste(overlay_img, (int(x_offset), int(y_offset)), overlay_img_mask)
        merge_image = Image.alpha_composite(merge_image, rendering_image)
        
    if args.debug and len(annotations) > 0 and i_entry < 10:    
        image_file = f"data/crello-v3/images/{template_id}_0.png"
        
        if (task == 'refine') or (task == 'complete'): # completion & refinement
            merge_image.convert('RGB').save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_{task}_input.jpg"))
        if not os.path.isfile(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_thumbnail.jpg")): # conditional
            thumbnail_img = Image.open(image_file).convert('RGB')
            thumbnail_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{template_id:08}_{page_num:01}_thumbnail.jpg"))
    return merge_image

def extract_elements(bbox_html):
    # Extract all bounding boxes using the provided pattern
    bbox_extract = re.compile(
             r'<rect\s+data-category="([^"]+)"\s*,\s*x="([^"]+)"\s*,\s*y="([^"]+)"\s*,\s*width="([^"]+)"\s*,\s*height="([^"]+)"\s*,\s*file_name="([^"]+)"\s*/>'
        )
    svg_content = re.search(r'<svg\s+([^>]+)>(.*?)<\/svg>', bbox_html, re.DOTALL)
    
    if not svg_content:
        print("No SVG content found")
        return None
    
    # Extract the attributes of the <svg> tag
    svg_attributes = svg_content.group(1)
    
    W = float(re.search(r'width=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
    H = float(re.search(r'height=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
    


    matches = bbox_extract.findall(bbox_html)
    
    # Find and return invalid elements
    invalid_elements, valid_elements = [], []
    for match in matches:
        category = match[0]
        x = float(match[1])
        y = float(match[2])
        w = float(match[3])
        h = float(match[4])
        file_name = match[5]

        if x < 0 or y < 0 or x+w > W or y+h > H:
            invalid_elements.append({
                "file_name": file_name,
                "data-category": category,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
            })
        else:
            valid_elements.append(file_name)

    return invalid_elements, valid_elements
    
def extract_unmasked_elements(bbox_html):
    bbox_extract = re.compile(
             r'<rect\s+data-category="([^"]+)"\s*,\s*x="([^"]+)"\s*,\s*y="([^"]+)"\s*,\s*width="([^"]+)"\s*,\s*height="([^"]+)"\s*,\s*file_name="([^"]+)"\s*/>'
        )
    matches = bbox_extract.findall(bbox_html)
    file_attributes = [
            {
                "file_name": match[5],
                "data-category": match[0],
                "x": match[1],
                "y": match[2],
                "width": match[3],
                "height": match[4],
            }
            for match in matches if all(attr != "<FILL_" for attr in match[1:5])
        ]
    return file_attributes

def modify_rects(input_string):
    # Function to process and potentially modify the float values
    def process_float(value):
        try:
            # Attempt to convert to float and then to integer
            return str(int(float(value)))
        except ValueError:
            # If the value is not a valid float, return it unchanged
            return value

    # Find all rect elements with their attributes
    rects = re.findall(
        r'(<rect[^>]*data-category="([^"]+)"[^>]*x="([^"]+)"[^>]*y="([^"]+)"[^>]*width="([^"]+)"[^>]*height="([^"]+)"[^>]*/>)',    
        input_string
    )

    for full_match, category, x, y, w, h in rects:
        # Process each float value
        new_x = process_float(x)
        new_y = process_float(y)
        new_w = process_float(w)
        new_h = process_float(h)

        # Reconstruct the rect element with the modified values
        new_rect = (
        f'<rect data-category="{category}" x="{new_x}" y="{new_y}" '
        f'width="{new_w}" height="{new_h}"/>'
        )

        # Replace the old rect element with the new one
        input_string = input_string.replace(full_match, new_rect)

    return input_string

def remove_elements(conversations, filenames):
    def remove_filenames(text):
        for filename_dict in filenames:
            text = re.sub(r'<rect\s+data-category="[^"]+"\s*,\s*x="[^"]+"\s*,\s*y="[^"]+"\s*,\s*width="[^"]+"\s*,\s*height="[^"]+"\s*,\s*file_name="' + re.escape(filename_dict['file_name']) + r'"\s*/>', '', text)
            # text = re.sub(r'<rect\s+data-category="[^"]+"\s*,\s*x1="[^"]+"\s*,\s*y1="[^"]+"\s*,\s*x2="[^"]+"\s*,\s*y2="[^"]+"\s*,\s*file_name="' + re.escape(filename_dict['file_name']) + r'"\s*/>', '', text)
        return text
    
    for conversation in conversations:
        if 'value' in conversation:
            conversation['value'] = remove_filenames(conversation['value'])
    return conversations

def merge_lists_without_overlap( list1, list2):
    # Use a dictionary to track unique dictionaries by file_name
    merged_dict = {d['file_name']: d for d in list1}
    
    # Update the dictionary with entries from the second list
    merged_dict.update({d['file_name']: d for d in list2})
    
    # Convert the dictionary back to a list of dictionaries
    merged_list = list(merged_dict.values())
    
    return merged_list

def main(args):
    # Model
    pattern = r', file_name="[^"]*"\s*'
    disable_torch_init()

    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.model_base, model_name)

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

    data = json.load(open(args.json_file, 'r', encoding='utf-8'))
    ret, gt = {}, {}
    
    os.makedirs(os.path.join('/'.join(args.output_file.split('/')[:-1]), 'gt'), exist_ok=True)
    os.makedirs(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{args.output_file.split('/')[-1].replace('.json', '')}"), exist_ok=True)
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    erase_file_name = r', file_name="[^"]*"\s*'
    
    for i_entry, entry in enumerate(tqdm(data, desc="Processing entries")):
        if entry['id'] not in ret:
            ret[entry['id']] = []
            gt[entry['id']] = []

        conv = conv_templates[args.conv_mode].copy()

        invalid_filenames, valid_filenames = extract_elements(entry['conversations'][1]['value'])
        merged_filelist = merge_lists_without_overlap(extract_unmasked_elements(entry['conversations'][0]['value']), invalid_filenames)

        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        template_id, page_num = entry['id'].split('_')
        page_num = int(page_num)
        
        if ('refine' in entry['image']) or ('complete' in entry['image']):
            image_file = f"crello-v3/images/{template_id}_1.png"
            image = online_rendering(args.data_path, image_file, merged_filelist, i_entry, args, entry['image']).convert('RGB')
        else:
            image_file = f"crello-v3/images/{template_id}_1.png"
            if len(invalid_filenames) > 0:
                image = online_rendering(args.data_path, image_file, invalid_filenames, i_entry, args, entry['image']).convert('RGB')
            else:
                image = Image.open(os.path.join(args.data_path, image_file)).convert('RGB')

        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)


        # preprocessing prompt
        ## 1. remove invalid elements
        entry['conversations']= remove_elements(entry['conversations'], invalid_filenames)
        
        ## 2. remove file_name
        entry['conversations'][0]['value'] = re.sub(erase_file_name, '', entry['conversations'][0]['value'])
        entry['conversations'][1]['value'] = re.sub(erase_file_name, '', entry['conversations'][1]['value'])   
        
        ## 3. modify float to integer
        entry['conversations'][0]['value'] = modify_rects(entry['conversations'][0]['value'])
        entry['conversations'][1]['value'] = modify_rects(entry['conversations'][1]['value'])

        inp = '\n'.join(entry["conversations"][0]['value'].split('\n')[1:])
        
        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            # image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = re.sub(pattern, '', prompt)
        
        if "posterllava/posterllava_v0" in args.model_path:
            prompt = replace_keys(prompt, MIRIDIH2QB)


        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        # gt[entry['id']].append(stringTojson_v2(entry['conversations'][1]['value']))
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                stopping_criteria=[stopping_criteria])
        
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs
        ret[entry['id']].append(stringTojson_v2(outputs))
        gt[entry['id']].append(stringTojson_v2(entry['conversations'][1]['value']))
        if args.debug or args.image_out:
            if ret[entry['id']][-1] == None:
                print("{} had output Nont". format(entry['id']))
            else:
                if not os.path.isfile(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"gt/{template_id:08}_{page_num:01}.jpg")):
                    try:
                        thumbnail_image_file = f"{args.data_path}/crello-v3/images/{template_id}_0.png"
                        thumbnail_img = Image.open(thumbnail_image_file).convert('RGB')
                        thumbnail_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"gt/{template_id:08}_{page_num:01}.jpg"))
                    except:
                        print("{} is not existing!!".format(f"{template_id:08}_{page_num:01}.jpg"))
                if entry['image'] == 'refine': # refinement
                    image_file = f"crello-v3/images/{template_id}_1.png"
                    if len(invalid_filenames) > 0:
                        args.debug = False
                        image = online_rendering(args.data_path, image_file, invalid_filenames, i_entry, args, entry['image']).convert('RGB')
                        args.debug = True
                    else:
                        image = Image.open(os.path.join(args.data_path, image_file)).convert('RGB')
                if "posterllava/posterllava_v0" in args.model_path:
                    drawn_img = draw_boxmap(ret[entry['id']][-1], valid_filenames, image, CLS2COLOR["QB"])  # Adjust the category as needed
                else:
                    drawn_img = draw_boxmap(ret[entry['id']][-1], valid_filenames, image, CLS2COLOR["Crello"])  # Adjust the category as needed
                drawn_img.save(os.path.join('/'.join(args.output_file.split('/')[:-1]), f"{args.output_file.split('/')[-1].replace('.json', '')}/{entry['id']}.jpg"))
                

    with open(args.output_file, 'w', encoding='utf-8') as fout:
        #json.dump(ret, fout, ensure_ascii=False, indent=2)
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
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--image-out", action="store_true")
    args = parser.parse_args()
    setproctitle('MIRIDIH-JJY')
    main(args)
