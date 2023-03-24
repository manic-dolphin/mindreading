import torch

from video_duma import *
from video_extract import *
from transformers import BertTokenizer, BertModel

frames = get_frames("/home/yyl/code/Labs/Data/video_raw/Clip2.mp4")
imgs_tensor = transform_frames(frames)

text = "I love nlp!"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
enc_text = tokenizer(text)
# print(enc_text)
inputs_ids = torch.tensor([enc_text['input_ids']])
attention_mask = torch.tensor([enc_text['attention_mask']])

model = V_DUMA()
logits = model(inputs_ids, attention_mask, imgs_tensor)
# model = BertModel.from_pretrained("bert-base-uncased")
# model(inputs_ids, attention_mask)
print(logits)