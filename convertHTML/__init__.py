from convertHTML.miridih import MiriDihDataset

def get_dataset(name, datapath, split, transform=None, min_size=[0,0], min_aspect_ratio=1e5, canvas_aspect_ratio=10, max_bbox_bins=32):
    if 'miridih' in name:
        return MiriDihDataset(datapath,split,max_seq_length=125,transform=transform, min_size=min_size, min_aspect_ratio=min_aspect_ratio, canvas_aspect_ratio=canvas_aspect_ratio)

    raise NotImplementedError(name)
