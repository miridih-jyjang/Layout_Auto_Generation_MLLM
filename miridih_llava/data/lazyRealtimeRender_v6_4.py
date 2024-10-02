from torch.utils.data import Dataset
import transformers, json, os, torch, copy
import sys
sys.path.append("/workspace/Layout_Auto_Generation_MLLM")  

from miridih_llava.data import DataArguments, rank0_print, preprocess_multimodal, preprocess
from typing import Dict
from PIL import Image
import re

class LazyRealTimeRenderingDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, ele_cache_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazyRealTimeRenderingDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.ele_cache_path = json.load(open(ele_cache_path, "r"))
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        # self.bbox_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([0-9.,\s]*)\],\s*'file_name':\s*'([^']*)'")
        self.bbox_extract =     re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)'")
        self.bbox_src_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
        self.bbox_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'file_name':\s*'([^']*)'")
        self.bbox_layer_IMG_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'\[IMG(\d+)\]':\s*'<image>',\s*'file_name':\s*'([^']*)'")
        self.bbox_src_layer_extract = re.compile(r"'label':\s*'([^']+)',\s*'box':\s*\[([-\d.,\s]*)\],\s*'layer':\s*(\d+),\s*'file_name':\s*'([^']*)',\s*'src':\s*'([^']*)'")
        self.erase_file_name =  r",\s*'file_name':\s*'[^']*'"
        # self.erase_image_token = "<image>.\n?"
        
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
        pixel_values = []
        assert (isinstance(i, int))
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            processor = self.data_args.image_processor
            invalid_filenames, valid_filenames = self.extract_invalid_elements(sources[0]['conversations'][1]['value'])

            for element in valid_filenames:
                ele_file = os.path.basename(element['file_name'])
                pixel_values.append(self.ele_cache_path[ele_file]) 
                
            if image_file == "complete" or image_file == "refine" or "coord_pred":
                template_id, page_num = self.list_data_dict[i]['id'].split('_')
                template_id = int(template_id)
                page_num = int(page_num)
                image_folder = self.data_args.image_folder
                if os.path.isfile(f"{image_folder}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                    image_file = f"{image_folder}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
                else:
                    image_file = f"{image_folder}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"

                
                merged_filelist = self.merge_lists_without_overlap(self.extract_unmasked_elements(sources[0]['conversations'][0]['value']), invalid_filenames)
                image = self.online_rendering(image_folder, image_file, merged_filelist).convert('RGB')
            
            else:                
                template_id, page_num = image_file.split('/')[-1].split('.')[0].split('_')
                template_id = int(template_id)
                page_num = int(page_num)
                image_folder = self.data_args.image_folder
                image_file = f"{image_folder}/miridih-v6.4/images/{template_id:08}/{page_num:03}/{os.path.basename(image_file)}"
                if len(invalid_filenames) > 0:
                    image = self.online_rendering(image_folder, image_file, invalid_filenames).convert('RGB')
                else:
                    
                    image = Image.open(image_file).convert('RGB')
                    
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

            # preprocessing prompt
            ## 1. remove invalid elements
            sources[0]['conversations']= self.remove_elements(sources[0]['conversations'], invalid_filenames)
            
            ## 2. remove file_name
            sources[0]['conversations'][0]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][0]['value'])
            # sources[0]['conversations'][0]['value'] = re.sub(self.erase_image_token, '', sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][1]['value'])   
            
            
            ## 3. change valid elementsvalid_filenamesvalid_filenames
            sources[0]['conversations'][0]['value'] = re.sub(r'(\d+)\sforeground elements', f'{len(valid_filenames)} foreground elements', sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][0]['value'] = re.sub(r'unordered\s(\d+) components', f'unordered {len(valid_filenames)} components', sources[0]['conversations'][0]['value'])
            
            # if len(pixel_values) > 0 and len(valid_filenames)+1 != sources[0]['conversations'][0]['value'].count('<image>'):
            #     print("task: {}\ttemplate: {}".format(self.list_data_dict[i]['image'], self.list_data_dict[i]['id']))
            #     sources[0]['conversations']= self.remove_elements(sources[0]['conversations'], invalid_filenames)

            sources = [e["conversations"] for e in sources]
            # sources = preprocess_multimodal(
            #     copy.deepcopy([e["conversations"] for e in sources]),
            #     self.data_args)
        else:
            # preprocessing prompt
            ## 1. remove invalid elements
            invalid_filenames, _ = self.extract_invalid_elements(sources[0]['conversations'][1]['value'])
            sources[0]['conversations']= self.remove_elements(sources[0]['conversations'], invalid_filenames)
            
            ## 2. remove file_name
            sources[0]['conversations'][0]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][1]['value']) 
           
            ## 3. change vali`d elements
            sources[0]['conversations'][0]['value'] = re.sub(r'(\d+)\sforeground elements', f'{len(valid_filenames)} foreground elements', sources[0]['conversations'][0]['value'])
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
        
        if len(pixel_values) == 0:
            crop_size = self.data_args.image_processor.crop_size
            data_dict['pixel_values'] = torch.empty(0)
        else:
            data_dict['pixel_values'] = pixel_values
        
        return data_dict

    def modify_rects(self, input_string):
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
            r"('label':\s*'([^']+)',\s*'box':\s*\[([0-9.,\s]*)\]')",
            input_string
        )
        for full_match, label, box in rects:
            # Process each float value
            new_x1 = process_float(box[0]) if box[0] != None else "''"
            new_y1 = process_float(box[1]) if box[0] != None else "''"
            new_x2 = process_float(box[2]) if box[0] != None else "''"
            new_y2 = process_float(box[3]) if box[0] != None else "''"

            # Reconstruct the rect element with the modified values
            new_rect = (f"'label': '{label}', 'box': [{new_x1}, {new_y1}, {new_x2}, {new_y2}]")

            # Replace the old rect element with the new one
            input_string = input_string.replace(full_match, new_rect)

        return input_string

    def online_rendering(self, image_folder, image_file, annotations):
        rendering_image = Image.open(os.path.join(image_folder, image_file))
        W, H = rendering_image.size
        merge_image = Image.new("RGBA", rendering_image.size)
        img_width, img_height = rendering_image.size
        annotations = sorted(annotations, key=lambda x: x['file_name'])

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

        # if len(annotations) > 0:
        #     temp= anno['file_name'].replace('.png', f'_{idx+1}.png')
        #     merge_image.save(f'/workspace/data/debugging/{temp}')
        
        #     image_file = f"/workspace/data/crello-v5/images/{template_id}_0.png"
                
        #     if not os.path.isfile(f"/workspace/data/debugging/gt/{template_id}_{page_num:01}_gt.png"):
        #         thumbnail_img = Image.open(image_file)
        #         thumbnail_img.save(f'/workspace/data/debugging/gt/{template_id}_{page_num:01}_gt.png')
        return merge_image

    def extract_unmasked_elements(self, bbox_html):
        matches = self.bbox_layer_IMG_extract.findall(bbox_html)
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
    
    def extract_invalid_elements(self, bbox_html):
        # Extract all bounding boxes using the provided pattern
        if 'src' in bbox_html:
            if 'layer' in bbox_html:
                matches = self.bbox_src_layer_extract.findall(bbox_html)
            else:
                matches = self.bbox_src_extract.findall(bbox_html)
        else:
            if 'layer' in bbox_html:
                matches = self.bbox_layer_extract.findall(bbox_html)
            else:
                matches = self.bbox_extract.findall(bbox_html)
        
        # Find and return invalid elements
        valid_elements = []
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

        return [], valid_elements

    def remove_elements(self, conversations, filenames):
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
    
    def merge_lists_without_overlap(self, list1, list2):
        # Use a dictionary to track unique dictionaries by file_name
        merged_dict = {d['file_name']: d for d in list1}
        
        # Update the dictionary with entries from the second list
        merged_dict.update({d['file_name']: d for d in list2})
        
        # Convert the dictionary back to a list of dictionaries
        merged_list = list(merged_dict.values())
        
        return merged_list

if __name__ == "__main__":

    class ModelArguments:
        model_name_or_path = "liuhaotian/llava-v1.5-7b"
        version = "v1"

    class TrainingArguments:
        cache_dir = None
        model_max_length = 4096

    class DataArguments:
        image_folder = "/workspace/data"
        image_processor = transformers.AutoFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14-336")
        image_aspect_ratio = "pad"
        is_multimodal = True
        mm_use_im_start_end = False

    model_args = ModelArguments()
    training_args = TrainingArguments()
    data_args = DataArguments()

    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token

    # Dummy conversation_lib for testing
    class ConversationLib:
        conv_templates = {"default": "Some template"}
        default_conversation = conv_templates["default"]

    conversation_lib = ConversationLib()

    # Path to the dataset JSON file
    data_path = "/workspace/data/miridih-v6.4/annotations/train_llava_numerical.json"
    # data_path = "/workspace/data/crello-v4/annotations/val_llava_numerical.json"

    # Initialize the dataset
    dataset = LazyRealTimeRenderingDataset(data_path, tokenizer, data_args)

    # Test the dataset
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)

    print("Dataset loaded and tested successfully.")
