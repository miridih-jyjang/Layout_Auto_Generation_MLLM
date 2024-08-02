import argparse
import torch
import json
import os
import re
from tqdm import tqdm

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from PIL import Image, ImageDraw

import requests
from io import BytesIO
from transformers import TextStreamer
CLS2COLOR = {
    "Miridih": {
        "generalsvg": "red", "text": "green", "shapesvg": "orange", "photo": "blue", "frameitem": "yellow",
        "lineshapeitem": "purple", "grid": "pink", "chart": "brown", "gif": "gray", "qrcode": "cyan", "video": "white",
        "barcode": "black", "youtube": "red", "basicsvg": "brown"
    },
    "QB-Poster": {
        "title": "red", "subtitle": "green", "itemlogo": "orange", "item": "blue", "itemtitle": "yellow",
        "object": "purple", "textbackground": "pink", "decoration": "brown", "frame": "gray", "text": "cyan",
        "false": "black"
    },
    "CGL": {
        "text": "red", "underlay": "green", "embellishment": "blue", "false": "black"
    },
    "Ad Banners": {
        "header": "red", "preheader": "green", "postheader": "blue", "body text": "orange", "disclaimer / footnote": "purple",
        "button": "pink", "callout": "brown", "logo": "gray", "false": "black"
    }
}

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
    drawn_outline = img.copy()
    drawn_fill = img.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)
    for cls, box in elems:
        color = cls2color[cls]
        left, top, right, bottom = box
        _box = int(left * W), int(top * H), int(right * W), int(bottom * H)
        draw_ol.rectangle(_box, fill=None, outline=color, width=max(10 * (W + H) // (1242 + 1660), 1))
        draw_f.rectangle(_box, fill=color)
    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.1))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    return drawn

def draw_boxmap(json_response, background_image, cls2color):
    pic = background_image.convert("RGB")
    cls_box = [(elem['label'], elem['box']) for elem in json_response]
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


def main(args):
    # Model
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

    for i_entry, entry in enumerate(tqdm(data, desc="Processing entries")):
        if entry['id'] not in ret:
            ret[entry['id']] = []
            gt[entry['id']] = []

        conv = conv_templates[args.conv_mode].copy()
        if "mpt" in model_name.lower():
            roles = ('user', 'assistant')
        else:
            roles = conv.roles

        image_file = os.path.join(args.data_path, entry['image'])
        image = load_image(image_file)
        # image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
        image_tensor = process_images([image], image_processor, args)
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)

        # inp = input(f"{roles[0]}: ")
        inp = '\n'.join(entry["conversations"][0]['value'].split('\n')[1:])
        
        # print(f"{roles[1]}: ", end="", flush=True)

        if image is not None:
            # first message
            if model.config.mm_use_im_start_end:
                inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
            else:
                inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
            conv.append_message(conv.roles[0], inp)
            image = None
        else:
            # later messages
            conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                streamer=streamer,
                use_cache=True,
                stopping_criteria=[stopping_criteria])

        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
        conv.messages[-1][-1] = outputs

        ret[entry['id']].append(stringTojson(outputs))

        if args.debug:
            gt[entry['id']].append(stringTojson(entry['conversations'][1]['value']))

            #print("\n", {"prompt": prompt, "outputs": outputs.split('\n')[0]},  "\n", flush=True)
            #print(entry["conversations"][1]['value'], flush=True)
            
            # Save images with boxes
            for json_response in ret[entry['id']]:
                img = load_image(os.path.join(args.data_path, [d['image'] for d in data if d['id'] == entry['id']][0]))
                drawn_img = draw_boxmap(json_response, img, CLS2COLOR["Miridih"])  # Adjust the category as needed
                drawn_img.save(os.path.join(args.data_path, f"{entry['id']}_boxed.jpg"))
            
            for json_response in gt[entry['id']]:
                img = load_image(os.path.join(args.data_path, [d['image'] for d in data if d['id'] == entry['id']][0]))
                drawn_img = draw_boxmap(json_response, img, CLS2COLOR["Miridih"])  # Adjust the category as needed
                drawn_img.save(os.path.join(args.data_path, f"{entry['id']}_gt.jpg"))

    with open(args.output_file, 'w', encoding='utf-8') as fout:
        json.dump(ret, fout, ensure_ascii=False, indent=2)

    

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
    args = parser.parse_args()
    main(args)
