import json
import random, re
import numpy as np
from matplotlib import pyplot as plt
import argparse, os, glob

def process_json(json_path):
    # # load template
    # data = open(template_path, "r").read()
    
    # load json
    with open(json_path, 'r') as file:
        content = file.read()

    # Split the content into individual JSON objects
    original_json_list = split_json_objects(content)
    print("total pages: ", len(original_json_list))
    out_json = []
    for original_json in original_json_list:
        # 00051725_0_1.png
        for key, value in original_json.items():     
            template_id = original_json["name"].split('/')[-1].split('_')[0]
            page_num = 0
            out_data = {}
            out_data["id"] = f"{template_id}_{page_num}"

            if key == 'name':
                continue
            ## online-rendering depends on tasks
            # if key == "cond_cate_to_size_pos_input_seqs" or key == "cond_cate_size_to_pos_input_seqs" or key == "cond_cate_pos_to_size_input_seqs" or \
            #     key == "unconditional_input_seqs" or key == "cond_recover_mask_input_seqs":
            #     if os.path.isfile(f"data/miridih/images134/{template_id:08}/{page_num:03}/images/{template_id:08}_{page_num:03}_skin.png"):
            #         out_data["image"] = f"miridih/images134/{template_id:08}/{page_num:03}/images/{template_id:08}_{page_num:03}_skin.png"
            #     else:
            #         out_data["image"] = f"miridih/images134/{template_id:08}/{page_num:03}/images/{template_id:08}_{page_num:03}_thumbnail.png"
            # elif key == "completion_input_seqs" or key == "refinement_input_seqs":
            #         out_data["image"] = "online_render"
            
            # else:
            #     continue
            if key == "coord_pred_seq_modeling":
                out_data["image"] = "coord_pred"
            elif key == "cond_cate_size_to_pos_seq_modeling" or key == "cond_cate_pos_to_size_seq_modeling" or \
                key == "cond_recover_mask_seq_modeling" or key == "cond_cate_size_to_pos_input_seqs" or key == "cond_cate_pos_to_size_input_seqs" or \
                key == "cond_recover_mask_input_seqs":
                out_data["image"] = f"crello-v6/images/{template_id}_1.png"
            elif key == "completion_seq_modeling" or key == "completion_input_seqs":
                out_data["image"] = "complete"
            elif key == "refinement_seq_modeling" or key == "refinement_seq_modeling" or \
            key == "completion_input_seqs" or key == "refinement_input_seqs":
                out_data["image"] = "refine"
            elif key == "coord_pred_seq_modeling":
                out_data["image"] = "coord_pred"
            else:
                continue
            
            parts = value.split("<MID>")
            # Extracting the <rect> elements and their attributes using regular expressions
        #     rects = re.findall(r'<rect[^>]*data-category="([^"]+)"[^>]*x="([^"]+)"[^>]*y="([^"]+)"[^>]*width="([^"]+)"[^>]*height="([^"]+)"', parts[0])
        #     # Create the JSON structure dynamically based on the number of <rect> elements
        #     json_output = []
        #     # Mapping the extracted values to the JSON template
        #     for i, rect in enumerate(rects):
        #         label, x, y, width, height = rect  # Extracting label, x, y, width, height
        #         x, y, width, height = map(float, [x, y, width, height])  # Convert strings to floats
        #         json_output.append({
        #             "label": label,
        #             "box": [x, y, width, height]
        #         })
            
        #     prompt = data.format(
        #     N=len(json_output),
        #     resolution=resolution,
        #     domain_name="social media promotion poster with qbposter style",
        #     json_data=json.dumps(json_output)
        # )
            if key == "completion_seq_modeling":
                parts[0] = parts[0].replace('background', 'incompleted')
                
            elif key == "refinement_seq_modeling":
                parts[0] = parts[0].replace('background', 'random distorted')
                parts[0] = parts[0].replace('completing', 'refining')
            
            if key == "coord_pred_seq_modeling":
                parts[0] = parts[0].replace('background image', 'background image <image0>')
                conversation = [
                    {"from": "human", "value": "Background image is <image0> <image>\n"+parts[0].strip(' ')},
                    {"from": "gpt", "value": "Sure! Here is the design results: " + parts[1].strip()} if len(parts) > 1 else {"from": "gpt", "value": ""}
                ]
            else:
                # add [IMG] to the string
                parts[0] = parts[0].replace('image ', 'image <image0> ')
                if key == "refinement_seq_modeling":
                    conversation = [
                        {"from": "human", "value": "Distorted image is <image0> <image>\n"+parts[0].strip(' ')},
                        {"from": "gpt", "value": "Sure! Here is the design results: " + parts[1].strip()} if len(parts) > 1 else {"from": "gpt", "value": ""}
                    ]
                elif key == "completion_seq_modeling":
                    conversation = [
                        {"from": "human", "value": "Incompleted image is <image0> <image>\n"+parts[0].strip(' ')},
                        {"from": "gpt", "value": "Sure! Here is the design results: " + parts[1].strip()} if len(parts) > 1 else {"from": "gpt", "value": ""}
                    ]
                else:
                    conversation = [
                        {"from": "human", "value": "Background image is <image0> <image>\n"+parts[0].strip(' ')},
                        {"from": "gpt", "value": "Sure! Here is the design results: " + parts[1].strip()} if len(parts) > 1 else {"from": "gpt", "value": ""}
                    ]

            out_data["conversations"] = conversation
            out_json.append(out_data)
    
    return out_json

def main(args):
    # process json
    json_path_list = glob.glob(os.path.join(args.json_path, "*.jsonl"))
    os.makedirs(args.output_dir, exist_ok=True)
    # json_path_list = [args.json_path + '/val_llama_numerical.jsonl']
    for json_path in json_path_list:
        data = process_json(json_path)

        # write in a json format
        if 'train' in json_path:
            output_path = args.output_dir + "train_llava_numerical.json"
        else:
            output_path = args.output_dir + "val_llava_numerical.json"
        # output_path = args.output_dir + "val_coord_pred.json"
        json.dump(data, open(output_path, 'w', encoding='utf-8'), indent=2)
        print(f"Data saved in {output_path}")
        

def split_json_objects(file_content):
    objects = []
    bracket_count = 0
    # idx = 0
    current_object = ""
    for char in file_content:
        if char == '{':
            bracket_count += 1
            current_object += char
        elif char == '}':
            bracket_count -= 1
            current_object += char
            # idx += 1
            # if idx > 100:
            #     break
            if bracket_count == 0:
                try:
                    # Attempt to parse the current object
                    json_object = json.loads(current_object)
                    objects.append(json_object)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON object: {current_object}")
                current_object = ""
        elif bracket_count > 0:
            current_object += char

    return objects

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="/workspace/data/crello-v6.7/html_format_2/")
    parser.add_argument("--train-image-root", type=str, default="/workspace/data/crello-v6/images")
    parser.add_argument("--val-image-root", type=str, default="/workspace/data/crello-v6/images")
    parser.add_argument("--output-dir", type=str, default="/workspace/data/crello-v6.7/annotations_2/")
    parser.add_argument("-d", type=int, default=4, help='round to D decimal places')
    args = parser.parse_args()
    main(args)
