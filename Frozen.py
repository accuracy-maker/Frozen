import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torchvision
import torchvision.transforms
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
import requests
from datasets import load_dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import T5Tokenizer, T5ForConditionalGeneration


# forward hook for reading resnet penultimate layer logits
def forward_hook(module, input, output):
    global resnet_avgpool_output
    resnet_avgpool_output = output

# class for image embeddings - obtained from resnet
class ImageEmbeddings(nn.Module):
    def __init__(self, hook_fn, d_model):
        super().__init__()
        self.resnet = torchvision.models.resnet50()
        self.resnet.avgpool.register_forward_hook(hook_fn)
        self.proj = nn.Linear(2048, d_model * 2, bias=False) # d_model * 2 because each image is supposed to constitute embeddings of seq_len = 2 (according to the paper)
    def forward(self, imgs): # imgs.shape: [b,c,w,h]
        batch_size = imgs.shape[0]
        _ = self.resnet(imgs)
        emb = resnet_avgpool_output # emb.shape: [b, 2048, 1, 1]
        emb = emb.flatten(start_dim=1, end_dim=-1) # emb.shape: [b, 2048]
        emb = self.proj(emb) # emb.shape: [b, d_model * 2]
        emb = emb.reshape(batch_size, d_model, 2)
        emb = emb.permute(0, 2, 1) # emb.shape: [batch_size, 2, d_model]
        return emb

def slice_datadict(dct, start_idx, end_idx):
    slice_dict = {}
    keys = list(dct.keys())
    for key in keys:
        slice_dict[key] = dct[key][start_idx:end_idx]
    return slice_dict

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

def calculate_loss(imgs,caption,image_encoder,t5_model,tokenizer,device):
    """Calculate Loss

    Parameters
    ----------
    imgs: torch.Tensor
        shape of imgs `[batch_size, channels, height, width]
    
    caption: Dict
        keys: input_ids, attn_mask
        values: torch.Tensor
    
    image_encoder: nn.Module

    t5_model: nn.Module
        Pre-Trained Language Model

    tokenizer: Hugging Face Tokenizer

    device: torch.device
        cuda or cpu

    Returns
    -------
    loss: float
    batch_accuracy: float
    """
    batch_size = imgs.shape[0]
    # obtain img embeddings
    img_embs = image_encoder(imgs) # img_embs.shape: [batch_size,2,d_model]
    # feed img_embs to t5 encoder to get encoder output
    enc_out = t5_model.encoder(
        inputs_embeds = img_embs
    ).last_hidden_state # enc_out.shape: [batch_size, 2, d_model]
    # extract cap tokens and attn_mask 
    cap_tokens, cap_attn_mask = caption.input_ids, caption.attention_mask
    # shift cap_tokens right (pre-pend start token) - as input to decoder is 
    # expected to be right shifted and starting with pad token (used as start token by T5)
    start_token_id = tokenizer(tokenizer.pad_token, 
                               return_tensors='pt', 
                               padding=False, truncation=True).input_ids
    start_token_id = start_token_id[:, 0] # trim end token appended by the tokenizer
    start_token_id = start_token_id.expand(batch_size, -1).to(device) # start_token_id.shape: [batch_size, 1]
    cap_tokens_rshifted = torch.cat((start_token_id, cap_tokens), dim=-1) # cap_tokens_rshifted.shape: [batch_size, seq_len+1]
    cap_tokens_rshifted = cap_tokens_rshifted[:, :-1] # cap_tokens_rshifted.shape: [batch_size, seq_len]

    # feed cap tokens to t5 decoder to get decoder output
    dec_out = t5_model.decoder(input_ids=cap_tokens_rshifted, 
                               attention_mask=cap_attn_mask, 
                               encoder_hidden_states=enc_out).last_hidden_state 
    # dec_out.shape: [batch_size, seq_len, d_model]

    # get scores from dec_out
    scores = t5_model.lm_head(dec_out) # scores.shape: [batch_size, seq_len, vocab_size]
    scores = scores.permute(0, 2, 1) # scores.shape: [batch_size, vocab_size, seq_len] - required for crossEntropyLoss

     # create targets = cap_tokens (unshifted)
    targets = cap_tokens # targets.shape: [batch_size, seq_len]

    # cross entropy loss 
    criterion = nn.CrossEntropyLoss(reduction='mean')
    loss = criterion(scores, targets)

    # calculate batch accuracy 
    pred_cap_tokens = torch.argmax(scores, dim=1) # shape: [batch_size, seq_len]
    batch_accuracy = (pred_cap_tokens == cap_tokens).float().mean() * 100
    return loss, batch_accuracy,targets,pred_cap_tokens

def ids2text(targets,pre_cap_tokens,tokenizer):
    """convert input_ids to caption

    Parameters
    ----------
    targets: torch.Tensor
        shape of targets `[batch_size,seq_len]`

    pre_cap_tokens: torch.Tensor
        shape of pre_cap_tokens `[batch_size,seq_len]`
    
    tokenizer: Hugging Face Tokenizer
        T5
    Returns
    -------
    True_Caption: List
    Pred_Caption: List
    """
    true_caption = []
    # true_caption = tokenizer.decode(
    #     targets,
    #     skip_special_tokens=True,
    # )
    batch = targets.shape[0]
    for i in range(batch):
        true_caption.append(tokenizer.decode(
            targets[i],
            skip_special_tokens=True,
        ))
    pred_caption = []
    for i in range(batch):
        pred_caption.append(tokenizer.decode(
            pred_cap_tokens[i],
            skip_special_tokens=True,
        ))
    return true_caption,pred_caption





## main ##
if __name__ == '__main__':
    # hyperparams
    img_size = 224 # resize for resnet
    d_model = 768 # d_model for T5 (required for resnet proj head)
    max_seq_len = 512 # required to init T5 Tokenizer
    batch_size = 16
    lr = 3e-4
    num_epochs = 10
    random_seed = 1010

    t5_model_name = 't5-base'

    checkpoint_path = 'ckpts_frozen_resnet/latest.pt' # path to a save and load checkpoint of the trained resnet
    resume_training_from_ckpt = False

    # set random seed
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    # cuda
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("mps")
    device = torch.device("cpu")


    # init image encoder model (resnet)
    image_encoder = ImageEmbeddings(forward_hook, d_model).to(device)

    # init T5 tokenizer and transformer model
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name, model_max_length=max_seq_len)
    t5_model = T5ForConditionalGeneration.from_pretrained(t5_model_name).to(device)

    dataset = load_dataset("ChristophSchuhmann/MS_COCO_2017_URL_TEXT")
    dataset = dataset['train'][:100]

    # init optimizer     
    optimizer = torch.optim.Adam(params=image_encoder.parameters(), lr=lr, betas=(0.9, 0.95))

    #train loop
    n = len(dataset["URL"])
    
    caption = []
    pre_caption = []


    log_dir = './data'
    writer = SummaryWriter(log_dir)
    for ep in tqdm(range(num_epochs)):
        Loss = []
        Acc = []
        for i in range(0, n, batch_size):
            print(f"epoch:{ep+1} | idx:{i}")
            sample = slice_datadict(dataset,i,i+batch_size)
            value_list = list(sample.values())
            url_list = value_list[0]
            captions = value_list[1]
            augmented_imgs = []

            #processing 
            img_list = []
            caption_list = []

            for url,_ in zip(url_list,captions):
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
            
            img_list.extend(augmented_imgs)
            caption_list.extend(captions)

            img = torch.stack(img_list, dim=0).to(device)
            cap = t5_tokenizer(caption_list, padding=True, truncation=True, return_tensors="pt")
            cap = cap.to(device)

            batch_loss, batch_accuracy,targets,pred_cap_tokens = calculate_loss(
                imgs=img,
                caption=cap,
                image_encoder=image_encoder,
                t5_model=t5_model,
                tokenizer=t5_tokenizer,
                device=device,
            )
            true_cap,pred_cap = ids2text(
                targets=targets,
                pre_cap_tokens=pred_cap_tokens,
                tokenizer=t5_tokenizer,
            )
            caption.extend(true_cap)
            pre_caption.extend(pred_cap)

            # update params
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            Loss.append(batch_loss.detach().cpu().item())
            Acc.append(batch_accuracy.detach().cpu().item())

        loss = np.mean(np.array(Loss))
        acc = np.mean(np.array(Acc))

        writer.add_scalar("loss",loss,ep)
        writer.add_scalar("accuracy",acc,ep)

    # print(f"caption: {caption}")
    # print(f"pred_caption: {pre_caption}")
    model_save_path = './ckpts_frozen_resnet/model_ep30.pth'
    torch.save(image_encoder.state_dict(), model_save_path)

    caption_array = np.array(caption)
    np.savetxt('./data/caption.txt',caption_array,fmt='%s' )
    pred_caption_array = np.array(pre_caption)
    np.savetxt('./data/pred_caption.txt',pred_caption_array,fmt='%s' )


