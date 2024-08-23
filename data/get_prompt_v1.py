import json
import random
import numpy as np
from matplotlib import pyplot as plt
from helper.global_var import *
import argparse, os, glob

def process_json(json_path):
    
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
            template_id, page_num, _ = original_json["name"].split("_")
            template_id = int(template_id)
            page_num = int(page_num)
            out_data = {}
            out_data["id"] = f"{template_id}_{page_num}"

            if key == 'name':
                continue
            ## online-rendering depends on tasks
            # if key == "cond_cate_to_size_pos_seq_modeling" or key == "cond_cate_size_to_pos_seq_modeling" or key == "cond_cate_pos_to_size_seq_modeling" or \
            #     key == "unconditional_seq_modeling" or key == "cond_recover_mask_seq_modeling" or key == "cond_cate_to_size_pos_input_seqs" or key == "cond_cate_size_to_pos_input_seqs" or key == "cond_cate_pos_to_size_input_seqs" or \
            #     key == "unconditional_input_seqs" or key == "cond_recover_mask_input_seqs":
            #     if os.path.isfile(f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
            #         out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
            #     else:
            #         out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"
            # elif key == "completion_seq_modeling" or key == "refinement_seq_modeling" or \
            # key == "completion_input_seqs" or key == "refinement_input_seqs":
            #         out_data["image"] = "online_render"

            if key == "cond_cate_size_to_pos_seq_modeling" or key == "cond_cate_pos_to_size_seq_modeling" or \
                key == "cond_recover_mask_seq_modeling" or key == "cond_cate_size_to_pos_input_seqs" or key == "cond_cate_pos_to_size_input_seqs" or \
                key == "cond_recover_mask_input_seqs":
                if os.path.isfile(f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                    out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
                else:
                    out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"
            elif key == "completion_seq_modeling" or key == "refinement_seq_modeling" or \
            key == "completion_input_seqs" or key == "refinement_input_seqs":
                    out_data["image"] = "online_render"

            else:
                continue

            # if key == "refinement_seq_modeling" or key == "refinement_input_seqs":
                # if os.path.isfile(f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                #     out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
                # else:
                #     out_data["image"] = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"
            #     out_data["image"] = "online_render"
            # else:
            #     continue

            parts = value.split("<MID>")
            if parts[0].endswith(".png"):
                print("wait")
            conversation = [
                {"from": "human", "value": parts[0].strip()},
                {"from": "gpt", "value": parts[1].strip()} if len(parts) > 1 else {"from": "gpt", "value": ""}
            ]

            out_data["conversations"] = conversation
            out_json.append(out_data)
    
    return out_json

def main(args):
    # process json
    json_path_list = glob.glob(os.path.join(args.json_path, "*.jsonl"))
    # json_path_list = [args.json_path + '/train_llama_numerical.jsonl']
    for json_path in json_path_list:
        data = process_json(json_path)

        # write in a json format
        if 'train' in json_path:
            output_path = args.output_dir + "train_llava_numerical.json"
        else:
            output_path = args.output_dir + "val_llava_numerical.json"
        # output_path = args.output_dir + "train_numerical_refine.json"
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
    parser.add_argument("--json-path", type=str, default="data/miridih/html_format/")
    parser.add_argument("--train-image-root", type=str, default="data/miridih/images")
    parser.add_argument("--val-image-root", type=str, default="data/miridih/images")
    parser.add_argument("--output-dir", type=str, default="data/miridih/annotations/")
    parser.add_argument("-d", type=int, default=4, help='round to D decimal places')
    args = parser.parse_args()
    main(args)

