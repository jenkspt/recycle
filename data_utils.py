import numpy as np
import json
from pathlib import Path
from urllib import parse
import requests

from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def download_data(save_dir):
    save_dir = Path(save_dir)
    assert save_dir.is_dir()
    # Annotations url
    # Validation
    url = "https://storage.googleapis.com/labelbox-exports/cjnxo84x0j9dh0797j4fnrf76/cjny1mlmn834y08762g947hm1/export-coco-2018-11-16T23%3A07%3A41.142812.json"
    # Training
    #url = "https://storage.googleapis.com/labelbox-exports/cjnxo84x0j9dh0797j4fnrf76/cjny1mlmn834y08762g947hm1/export-coco-2018-11-02T21%3A43%3A32.227623.json"

    # Save annotations
    r = requests.get(url, allow_redirects=True)
    data = r.json()
    data = fix_image_ids(data)

    for image_data in data['images']:
        parsed_url = parse.urlparse(parse.unquote(image_data['file_name']))
        file_name = Path(parsed_url.path).name
        image_data['file_name'] = file_name
    json.dump(data, (save_dir / 'valid.json').open('w'))

def fix_image_ids(data):
    """ 
    LabelBox uses strings for image ids which don't work
    with the coco API (which is used inside of torchvision CocoDetection
    dataset class 
    """
    img_ids = dict()
    for i, img_data in enumerate(data['images']):
        id = img_data['id']
        # Create lookup table
        img_ids[id] = i
        # Set image id to new integer id
        img_data['id'] = i

    for ann in data['annotations']:
        # Set image id to new integer id for each annotation
        ann['image_id'] = img_ids[ann['image_id']]
    return data

def centered_crop(image, x,y,w,h):
    """ PIL Image """
    return image.crop((x-w//2, y-w//2, x+np.ceil(w/2), y+np.ceil(h/2)))

def make_classify_dataset(base_dir, save_dir, crop_size):
    """ """
    base_dir = Path(base_dir)
    assert base_dir.is_dir()
    save_dir = Path(save_dir)
    assert save_dir.is_dir()
    
    class_dict = {1:'glass',2:'metal',3:'plastic'}

    for name in ('valid', 'train'):
        ds = CocoDetection(
                root = str(base_dir / name),
                annFile = str(base_dir / f'{name}.json'),
                transform=None)
         
        for img, labels in ds:
            for label in labels:
                x,y,w,h = label['bbox']
                cx, cy = x + w//2, y + h//2
                crop = centered_crop(img, cx, cy, crop_size, crop_size)
                # Get class name for directory structure
                class_name = class_dict[label['category_id']]
                # Use annotation id for image name
                save_path = save_dir / name / class_name
                save_path.mkdir(parents=True, exist_ok=True)
                crop.save(save_path / f'{label["image_id"]}_{label["id"]}.jpg')

def display(image, annotations, ax=None):
    """
    PIL, lst
    """
    if ax == None:
        fig, ax = plt.subplots(1)

    ax.imshow(image)
    class_colors = {1:'#169487', 2:'#7CB14A', 3:'#B19CD9'}
    for ann in annotations:
        x, y, w, h = ann['bbox']
        color = class_colors[ann['category_id']]
        rect_border = patches.Rectangle((x,y), w, h, 
                linewidth=2, edgecolor=color, facecolor='none', alpha=.6)

        rect_fill = patches.Rectangle((x,y), w, h, 
                linewidth=0, edgecolor='none', facecolor=color, alpha=.2)
        ax.add_patch(rect_border)
        ax.add_patch(rect_fill)
    ax.axis('off')
    _glass = patches.Patch(color='#169487', label='Glass')
    _metal = patches.Patch(color='#7CB14A', label='Metal')
    _plastic = patches.Patch(color='#B19CD9', label='Plastic')
    ax.legend(handles=[_glass, _metal, _plastic])
    return ax

if __name__ == "__main__":
    pass
    #save_dir = 'data/recycle_coco'
    #download_data(save_dir)
    #make_classify_dataset('data/recycle_coco', 'data/recycle_classify',447)
    make_classify_dataset('data/recycle_coco', 'data/recycle_classify',632)
    """
    ds = CocoDetection(
        root = 'data/recycle_coco/train',
        annFile = 'data/recycle_coco/train.json',
        transform=None)

    display(*ds[0])
    plt.show()
    """
