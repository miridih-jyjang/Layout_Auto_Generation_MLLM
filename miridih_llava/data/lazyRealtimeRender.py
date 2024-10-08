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

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazyRealTimeRenderingDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.bbox_extract = re.compile(
             r'<rect\s+data-category="([^"]+)"\s*,\s*x="([^"]+)"\s*,\s*y="([^"]+)"\s*,\s*width="([^"]+)"\s*,\s*height="([^"]+)"\s*,\s*file_name="([^"]+)"\s*/>'
        )
        self.erase_file_name = r', file_name="[^"]*"\s*'
        
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
        assert (isinstance(i, int))
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            processor = self.data_args.image_processor
            invalid_filenames = self.extract_invalid_elements(sources[0]['conversations'][1]['value'])
            
            if image_file == "online_render":
                template_id, page_num = self.list_data_dict[i]['id'].split('_')
                template_id = int(template_id)
                page_num = int(page_num)
                # miridih/images/00049817/000/00049817_0_1.png
                if os.path.isfile(f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                    image_file = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
                else:
                    image_file = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"

                merged_filelist = self.merge_lists_without_overlap(self.extract_unmasked_elements(sources[0]['conversations'][0]['value']), invalid_filenames)
                image = self.online_rendering(image_folder, image_file, merged_filelist).convert('RGB')
                
            else:
                
                if len(invalid_filenames) > 0:
                    template_id, page_num = self.list_data_dict[i]['id'].split('_')
                    template_id = int(template_id)
                    page_num = int(page_num)
                    if os.path.isfile(f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"):
                        image_file = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_1.png"
                    else:
                        image_file = f"miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"

                    image = self.online_rendering(image_folder, image_file, invalid_filenames).convert('RGB')
                else:
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

            # preprocessing prompt
            ## 1. remove invalid elements
            sources[0]['conversations']= self.remove_elements(sources[0]['conversations'], invalid_filenames)
            
            ## 2. remove file_name
            sources[0]['conversations'][0]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][1]['value'])   
            
            ## 3. modify float to integer
            sources[0]['conversations'][0]['value'] = self.modify_rects(sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = self.modify_rects(sources[0]['conversations'][1]['value'])
            
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            # preprocessing prompt
            ## 1. remove invalid elements
            invalid_filenames = self.extract_invalid_elements(sources[0]['conversations'][1]['value'])
            sources[0]['conversations']= self.remove_elements(sources[0]['conversations'], invalid_filenames)
            
            ## 2. remove file_name
            sources[0]['conversations'][0]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = re.sub(self.erase_file_name, '', sources[0]['conversations'][1]['value']) 
           
            ## 3. modify float to int
            sources[0]['conversations'][0]['value'] = self.modify_rects(sources[0]['conversations'][0]['value'])
            sources[0]['conversations'][1]['value'] = self.modify_rects(sources[0]['conversations'][1]['value'])
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
        # rects = re.findall(
        #     r'(<rect[^>]*data-category="([^"]+)"[^>]*x1="([^"]+)"[^>]*y1="([^"]+)"[^>]*x2="([^"]+)"[^>]*y2="([^"]+)"[^>]*/>)',
        #     input_string
        # )
    
        # for full_match, category, x1, y1, x2, y2 in rects:
        #     # Process each float value
        #     new_x1 = process_float(x1)
        #     new_y1 = process_float(y1)
        #     new_x2 = process_float(x2)
        #     new_y2 = process_float(y2)

        #     # Reconstruct the rect element with the modified values
        #     new_rect = (
        #     f'<rect data-category="{category}" x1="{new_x1}" y1="{new_y1}" '
        #     f'x2="{new_x2}" y2="{new_y2}"/>'
        #     )

            # Replace the old rect element with the new one
            input_string = input_string.replace(full_match, new_rect)

        return input_string

    def online_rendering(self, image_folder, image_file, annotations):
        rendering_image = Image.open(os.path.join(image_folder, image_file))
        merge_image = Image.new("RGBA", rendering_image.size)
        img_width, img_height = rendering_image.size
        annotations = sorted(annotations, key=lambda x: x['file_name'])

        for idx, anno in enumerate(annotations):
            ele_img_file = anno['file_name']
            template_id, page_num, _ = ele_img_file.split('_')
            template_id = int(template_id)
            page_num = int(page_num)
            image_file = f"{image_folder}/miridih/images/{template_id:08}/{page_num:03}/{ele_img_file}"
            overlay_img = Image.open(image_file).convert("RGBA")
            ele_width, ele_height = float(anno['width']), float(anno['height'])
            
            x_offset, y_offset = float(anno['x']), float(anno['y'])
            overlay_img = overlay_img.resize((max(1, round(ele_width)), max(1,round(ele_height))))
            _, _, _, overlay_img_mask = overlay_img.split()
            rendering_image.paste(overlay_img, (int(x_offset), int(y_offset)), overlay_img_mask)
            merge_image = Image.alpha_composite(merge_image, rendering_image)

        # if len(annotations) > 0:
        #     temp= anno['file_name'].replace('.png', f'_{idx+1}.png')
        #     merge_image.save(f'{image_folder}/online_render_skin/{temp}')
        
        #     image_file = f"data/miridih/images/{template_id:08}/{page_num:03}/{template_id:08}_{page_num:01}_0.png"
                
        #     if not os.path.isfile(f"data/online_render_skin/{template_id:08}_{page_num:01}_gt.png"):
        #         thumbnail_img = Image.open(image_file)
        #         thumbnail_img.save(f'data/online_render_skin/{template_id:08}_{page_num:01}_gt.png')
        return merge_image

    def extract_unmasked_elements(self, bbox_html):
        matches = self.bbox_extract.findall(bbox_html)
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
    
    def extract_invalid_elements(self, bbox_html):
        # Extract all bounding boxes using the provided pattern
        matches = self.bbox_extract.findall(bbox_html)
        svg_content = re.search(r'<svg\s+([^>]+)>(.*?)<\/svg>', bbox_html, re.DOTALL)
    
        if not svg_content:
            print("No SVG content found")
            return None
        
        # Extract the attributes of the <svg> tag
        svg_attributes = svg_content.group(1)
        
        W = float(re.search(r'width=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
        H = float(re.search(r'height=["\'](\d+(?:\.\d+)?)["\']', svg_attributes).group(1))
        
        # Find and return invalid elements
        invalid_elements = []
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

        return invalid_elements

    def remove_elements(self, conversations, filenames):
        def remove_filenames(text):
            for filename_dict in filenames:
                text = re.sub(r'<rect\s+data-category="[^"]+"\s*,\s*x="[^"]+"\s*,\s*y="[^"]+"\s*,\s*width="[^"]+"\s*,\s*height="[^"]+"\s*,\s*file_name="' + re.escape(filename_dict['file_name']) + r'"\s*/>', '', text)
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
        image_folder = "./data"
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
    data_path = "data/miridih/annotations/train_llava_numerical.json"

    # Initialize the dataset
    dataset = LazyRealTimeRenderingDataset(data_path, tokenizer, data_args)

    # Test the dataset
    for i in range(len(dataset)):
        sample = dataset[i]
        print(sample)

    print("Dataset loaded and tested successfully.")
