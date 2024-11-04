import json
import os
import glob
import argparse

def process_json(json_path):
    with open(json_path, 'r') as file:
        content = file.read()

    original_json_list = split_json_objects(content)
    print("Total pages: ", len(original_json_list))

    # Dictionary to hold data for each task
    task_data = {
        "val_random": [],
        "val_refine": [],
        "val_cp2s": [],
        "val_cs2p": [],
        "val_complete": [],
        "val_coord_pred": []
    }
    out_json = []
    for original_json in original_json_list:
        for key, value in original_json.items():
            template_id = original_json["name"].split('/')[-1].split('_')[0]
            page_num = 0
            out_data = {"id": f"{template_id}_{page_num}"}

            if key == 'name':
                continue
            if 'train' in json_path:
                if key in ["cond_cate_size_to_pos_seq_modeling", "cond_cate_size_to_pos_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                elif key in ["cond_cate_pos_to_size_seq_modeling", "cond_cate_pos_to_size_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                elif key in ["cond_recover_mask_seq_modeling", "cond_recover_mask_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                elif key in ["completion_seq_modeling", "completion_input_seqs"]:
                    out_data["image"] = "complete"
                elif key in ["refinement_seq_modeling", "refinement_seq_modeling"]:
                    out_data["image"] = "refine"
                elif key == "coord_pred_seq_modeling":
                    out_data["image"] = "coord_pred"
                else:
                    continue    
                
            else:
                if key in ["cond_cate_size_to_pos_seq_modeling", "cond_cate_size_to_pos_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                    task_data["val_cs2p"].append(out_data)
                elif key in ["cond_cate_pos_to_size_seq_modeling", "cond_cate_pos_to_size_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                    task_data["val_cp2s"].append(out_data)
                elif key in ["cond_recover_mask_seq_modeling", "cond_recover_mask_input_seqs"]:
                    out_data["image"] = f"crello-v6.7/images/{template_id}_1.png"
                    task_data["val_random"].append(out_data)
                elif key in ["completion_seq_modeling", "completion_input_seqs"]:
                    out_data["image"] = "complete"
                    task_data["val_complete"].append(out_data)
                elif key in ["refinement_seq_modeling", "refinement_seq_modeling"]:
                    out_data["image"] = "refine"
                    task_data["val_refine"].append(out_data)
                elif key == "coord_pred_seq_modeling":
                    out_data["image"] = "coord_pred"
                    task_data["val_coord_pred"].append(out_data)
                else:
                    continue

            # Prepare conversation structure (simplified for this example)
            parts = value.split("<MID>")
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
            if 'train' in json_path:
                out_json.append(out_data)
                
    if 'train' in json_path:
        return out_json
    else:
        return task_data

def main(args):
    json_path_list = glob.glob(os.path.join(args.json_path, "*.jsonl"))
    os.makedirs(args.output_dir, exist_ok=True)

    for json_path in json_path_list:
        data = process_json(json_path)

        if 'train' in json_path:
            output_path = os.path.join(args.output_dir, "train_llava_numerical.json")
            json.dump(data, open(output_path, 'w', encoding='utf-8'), indent=2)
            print(f"Data saved in {output_path}")
        else:
            output_files = {
                "val_random": "val_random.json",
                "val_refine": "val_refine.json",
                "val_cp2s": "val_cp2s.json",
                "val_cs2p": "val_cs2p.json",
                "val_complete": "val_complete.json",
                "val_coord_pred": "val_coord_pred.json"
            }

            for task, file_name in output_files.items():
                output_path = os.path.join(args.output_dir, file_name)
                json.dump(data[task], open(output_path, 'w', encoding='utf-8'), indent=2)
                print(f"Data saved in {output_path}")

def split_json_objects(file_content):
    objects = []
    bracket_count = 0
    current_object = ""
    for char in file_content:
        if char == '{':
            bracket_count += 1
            current_object += char
        elif char == '}':
            bracket_count -= 1
            current_object += char
            if bracket_count == 0:
                try:
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
    parser.add_argument("--output-dir", type=str, default="/workspace/data/crello-v6.7/annotations_2/")
    args = parser.parse_args()
    main(args)
