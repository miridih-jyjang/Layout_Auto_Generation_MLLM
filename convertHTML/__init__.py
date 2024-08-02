from convertHTML.miridih import MiriDihDataset

def get_dataset(name, datapath, split, transform=None, max_bbox_bins=32):
    if name == "miridih":
        return MiriDihDataset(datapath,split,max_seq_length=125,transform=transform)
    
    raise NotImplementedError(name)
