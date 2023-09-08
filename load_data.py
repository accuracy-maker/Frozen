import os
import requests
import cv2
import numpy as np
import torch
import torchvision.transforms
from datasets import load_dataset
from transformers import T5Tokenizer

# def slice_dict(dct, start_idx, end_idx):
#     keys = list(dct.keys())
#     sliced_keys = keys[start_idx:end_idx]
#     sliced_dict = {k: dct[k] for k in sliced_keys}
#     return sliced_dict
def slice_datadict(dct, start_idx, end_idx):
    slice_dict = {}
    keys = list(dct.keys())
    for key in keys:
        slice_dict[key] = dct[key][start_idx:end_idx]
    return slice_dict

# 加载数据集
dataset = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT")

# 获取训练数据集
train_dataset = dataset["train"]

# 选择前100张图片进行测试
test_dataset = train_dataset[:30]

# 初始化分词器
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# 定义图像尺寸
img_size = 224  # 或适合您模型的图像尺寸

# 定义设备（GPU或CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_batch_(minibatch, img_size):
    """process the url

    Parameters
    ----------
    minibatch: Dict
        key: ['URL','text']
        value: list[URL],list[text]

    img_size: int
        the size of image
    
    
    Returns
    -------
    augmented_imgs: List
        length of augmented_imgs: batch
    captions: List
        length of caption: batch
    """
    value_list = list(minibatch.values())
    url_list = value_list[0]
    captions = value_list[1]
    augmented_imgs = []
    #processing 
    for url,cap in zip(url_list,captions):
        print(f"processing url: {url}")
        # print(f"caption: {cap}")
        response = requests.get(url)
        if response.status_code == 200:
            img_data = response.content
        else:
            print(f"Failed to fetch image from URL. Status code: {response.status_code}")
            continue 
        # img_data = response.content
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), -1)
        resize_shape = (img_size, img_size)
        img = cv2.resize(img, resize_shape, interpolation=cv2.INTER_LINEAR)
        img = np.float32(img) / 255
        img = torch.tensor(img)
        img = img.permute(2, 1, 0)  # [w, h, c] -> [c, h, w]
        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(int(1.25 * img_size)),  # image_size + 1/4 * image_size
            torchvision.transforms.RandomResizedCrop(resize_shape, scale=(0.8, 1.0)),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # zero mean, unit std
        ])
        img = transforms(img)
        augmented_imgs.append(img)
        
    return augmented_imgs, captions





def load_img2cap(batch_size,dataset,tokenizer,img_size, device):
    """load the image-caption dataset and return torch.Tensor

    Parameters
    ----------
    batch_size: int

    dataset: DataDict

    tokenizer: T5

    device: string -- torch.device
        cuda or cpu

    Returns
    -------
    img_tenosr: torch.Tensor
    caption_tenosr: torch.Tensor

    """
    img_list = []
    caption_list = []
    n = len(list(dataset.values())[0])
    for i in range(0, n, batch_size):
        minibatch = slice_datadict(dataset, i, i+batch_size)
        augmented_imgs, captions = process_batch_(minibatch, img_size)
        img_list.extend(augmented_imgs)
        caption_list.extend(captions)
        print("-----------------------")
    
    img_tensor = torch.stack(img_list, dim=0).to(device)
    caption = tokenizer(caption_list, padding=True, truncation=True, return_tensors="pt")
    # caption = caption.to(device)
    # caption = {key: val.to(device) for key, val in caption.items()}
    caption = {
    key: val.to(device) if isinstance(val, torch.Tensor) else val
    for key, val in tokenizer(caption_list, padding=True, truncation=True, return_tensors="pt").items()
}
    return img_tensor,caption

img, caption = load_img2cap(
    batch_size=5,
    dataset=test_dataset,
    tokenizer=tokenizer,
    img_size=img_size,
    device=device,
)
input_ids = caption['input_ids']

# 打印张量的形状
print("Image Tensor Shape:", img.shape)
print("token id Tensor Shape:", input_ids.shape)
