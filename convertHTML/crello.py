import pandas as pd
import torch
import os, glob
from fsspec.core import url_to_fs
from torch_geometric.data import Data
from convertHTML.base import BaseDataset
from helper.global_var import *

class CrelloDataset(BaseDataset):
    name = "crello"
    labels =[
        "svgElement",
        "textElement",
        "imageElement",
        "coloredBackground",
        "maskElement"
    ]
    
    def __init__(self,dir,split,max_seq_length,transform=None, min_size=[0,0], min_aspect_ratio=5, canvas_aspect_ratio=10):
        self.min_size = min_size
        self.min_aspect_ratio = min_aspect_ratio
        self.canvas_aspect_ratio = canvas_aspect_ratio
        super().__init__(dir,split,transform)
        self.N_category = self.num_classes
        self.dataset_name = "crello"
        self.transform = transform
    

    def process(self):
        if not os.path.isfile(self.processed_paths[0]):
            

            # Load CSV file
            fs,_=url_to_fs(self.raw_dir)
            dir_list = glob.glob(os.path.join(self.raw_dir,"*"))
            for d in dir_list:
                csv_file_list = glob.glob(os.path.join(d, "*"))
                data_list = []
                for csv_file in csv_file_list:
                    data = pd.read_csv(csv_file)
                    # data['priority'].replace({'Thumbnail_Images': -1, 'Skin_Images': -1}, inplace=True)

                    data.loc[:, 'priority'] = data['priority'].astype(float).astype(int)
                    data = data.sort_values(by='priority')

                    for (template_idx, page_num), group in data.groupby(['template_idx', 'page_num']): 
                        box, label, text, attr, opacity, rot, file_names, priority = [], [], [], {}, [], [], [], []
                        is_valid = True
                        for _, row in group.iterrows():
                            W, H = row['canvas_width'], row['canvas_height']
                            attr = {
                                "name": row['reformat_image_file_name'],
                                "width": row['canvas_width'],
                                "height": row['canvas_height'],
                                "filtered": False,
                                "has_canvas_element": False,
                                "NoiseAdded": False,
                                "template_type": row['super_template_type']
                            }
                            if W/H > self.canvas_aspect_ratio or W/H < 1/self.canvas_aspect_ratio:
                                is_valid = False
                                break

                            category = row['tag']
                            
                            # W, H = 1920, 1080 # T.B.D.
                            if category in self.labels:
                                cat_id = self.labels.index(category)+1  # Assuming all elements have category_id 1 for this example
                            else:
                                cat_id = 0
                            
                            if float(row['img_width']) <= self.min_size[0] or float(row['img_height']) <= self.min_size[1]:
                                continue
                            
                            if float(row['img_width']) / float(row['img_height']) > self.min_aspect_ratio or float(row['img_width']) / float(row['img_height']) < 1/self.min_aspect_ratio:
                                is_valid = False
                                break

                            elements = [{
                                'bbox': [float(row['left']), float(row['top']), float(row['img_width']), float(row['img_height'])],
                                'text': row.get('text_content', None),
                                'category_id': cat_id,
                                'opacity': row['opacity'],
                                'rotation': row['rotation'],
                                'file_name': row['reformat_image_file_name'], 
                                'priority': row['priority']
                            }]
                            
                            for element in elements:
                                bbox = element['bbox']
                                x_m, y_m, w, h = bbox
                                xc = (x_m + w / 2)
                                yc = (y_m + h / 2)
                                
                                b = [round(xc / W, 2), round(yc / H, 2), round(w / W, 2), round(h / H, 2)]
                                
                                cat = element['category_id']
                                te = element.get('text', None)
                                z = element['priority']
                                
                                box.append(b)
                                label.append(cat)
                                opacity.append(element['opacity'])
                                rot.append(element['rotation'])
                                file_names.append(element['file_name'])
                                priority.append(z)
                                
                                if type(te) == float:
                                    text.append(None)
                                else:
                                    text.append(te)

                        if is_valid:
                            data = Data(x=torch.tensor(box,dtype=torch.float),y=torch.tensor(label,dtype=torch.long),
                                        text = text, opacity=opacity, rotation=rot, file_name=file_names, priority=priority)
                            data.attr=attr

                            data_list.append(data)
            
                generator = torch.Generator().manual_seed(0)
                indices = torch.randperm(len(data_list), generator=generator)  # shuffling
                data_list = [data_list[i] for i in indices]
                N = len(data_list)
                print(N)
                #s = [int(N * 0.8), int(N * 0.9)]
            
                # Save processed data
                os.makedirs('processed', exist_ok=True)
                # torch.save(data_list[:s[0]], 'processed/train.pt')
                # torch.save(data_list[s[0]:s[1]], 'processed/val.pt')
                # torch.save(data_list[s[1]:], 'processed/test.pt')

        #        with fs.open(self.processed_paths[0], "wb") as file_obj:
        #          torch.save(self.collate(data_list[: s[0]]), file_obj)
        #     with fs.open(self.processed_paths[1], "wb") as file_obj:
            #        torch.save(self.collate(data_list[s[0] : s[1]]), file_obj)
            #   with fs.open(self.processed_paths[2], "wb") as file_obj:
            #      torch.save(self.collate(data_list[s[1] :]), file_obj)
                if 'train' in d:
                    with fs.open(self.processed_paths[0], "wb") as file_obj:
                        torch.save(self.collate(data_list), file_obj)
                else:
                    with fs.open(self.processed_paths[1], "wb") as file_obj:
                        torch.save(self.collate(data_list), file_obj)
    # Example usage
    # dataset = MiridihDataset(['/home/jang/Documents/code/Miridih/data/metadata_elements_train_1.csv',
    #                             '/home/jang/Documents/code/Miridih/data/metadata_elements_train_2.csv',
    #                             '/home/jang/Documents/code/Miridih/data/metadata_elements_train_3.csv'])
    # dataset.process()


