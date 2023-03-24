import pandas as pd
import numpy as np
from video_extract import *
import re
import os

def get_data(frames):
    
    file_dir = os.getcwd()
    # data_sf1 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_1_Text.xlsx")
    data_sf1 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_1_Text.xlsx'))
    data_sf1['Question'] = "Why do you think the men hide?"
    data_sf1['Frames'] = [frames[0]] * len(data_sf1)

    # data_sf2 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_2_Text.xlsx")
    data_sf2 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_2_Text.xlsx'))
    data_sf2['Question'] = "What do you think the woman is thinking?"
    data_sf2['Frames'] = [frames[1]] * len(data_sf2)

    # data_sf3 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_3_Text.xlsx")
    data_sf3 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_3_Text.xlsx'))
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"
    data_sf3['Frames'] = [frames[2]] * len(data_sf3)

    # data_sf4 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_4_Text.xlsx")
    data_sf4 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_4_Text.xlsx'))
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"
    data_sf4['Frames'] = [frames[3]] * len(data_sf4)

    # data_sf5 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_5_Text.xlsx")
    data_sf5 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_5_Text.xlsx'))
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"
    data_sf5['Frames'] = [frames[4]] * len(data_sf5)

    # data_sf6 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_6_Text.xlsx")
    data_sf6 = pd.read_excel(os.path.join(file_dir, 'Data/relabel/SFQuestion_6_Text.xlsx'))
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"
    data_sf6['Frames'] = [frames[5]] * len(data_sf6)

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Frames', 'Question', 'Answer', 'Score'])

    dataset['Frames'] = data_all_raw['Frames']
    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset


def get_dataset_bert():
    data_sf1 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_1_Text.xlsx")
    data_sf1['Question'] = "Why do you think the men hide?"

    data_sf2 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_2_Text.xlsx")
    data_sf2['Question'] = "What do you think the woman is thinking?"

    data_sf3 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_3_Text.xlsx")
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"

    data_sf4 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_4_Text.xlsx")
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"

    data_sf5 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_5_Text.xlsx")
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"

    data_sf6 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_6_Text.xlsx")
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Question', 'Answer', 'Score'])

    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset

def get_data_baseline():
    data_sf1 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_1_Text.xlsx")
    data_sf1['Question'] = "Why do you think the men hide?"

    data_sf2 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_2_Text.xlsx")
    data_sf2['Question'] = "What do you think the woman is thinking?"

    data_sf3 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_3_Text.xlsx")
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"

    data_sf4 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_4_Text.xlsx")
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"

    data_sf5 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_5_Text.xlsx")
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"

    data_sf6 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_6_Text.xlsx")
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Question', 'Answer', 'Score'])

    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset

def get_SSdataset():
    #substitute the file path with your own.
    #this path is where I store the data.
    data_ss_brain = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SS_Brian_Text.xlsx")
    data_ss_brain[
        'Passage'] = "Brian is always hungry. Today at school it is his favourite meal–sausages and beans. He is a very greedy boy, and he would like to have more sausages than anybody else, even though his mother will have made him a lovely meal when he gets home! But everyone is allowed two sausages and no more. When it is Brian’s turn to be served, he says, ‘Oh please can I have four sausages because I won’t be having any dinner when I get home!'[SEP]"
    data_ss_brain['Question'] = "Why does Brian say this?"

    data_ss_burglar = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SS_Burglar_Text.xlsx")
    data_ss_burglar[
        'Passage'] = "A burglar who has just robbed a shop is making his getaway. As he is running home, a policeman on his beat sees him drop his glove. He doesn’t know the man is a burglar, he just wants to tell him he dropped his glove. But when the policeman shouts out to the burglar, ‘Hey, you! Stop!’, the burglar turns round, sees the policeman and gives himself up. He puts his hands up and admits that he did the break-in at the local shop.[SEP]"
    data_ss_burglar['Question'] = "Why did the burglar do that?"

    data_ss_peabody = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SS_Peabody_Text.xlsx")
    data_ss_peabody[
        'Passage'] = "Late one night old Mrs Peabody is walking home. She doesn’t like walking home alone in the dark because she is always afraid that someone will attack her and rob her. She really is a very nervous person! Suddenly, out of the shadows comes a man. He wants to ask Mrs Peabody what time it is, so he walks toward her. When Mrs Peabody sees the man coming toward her, she starts to tremble and says, ‘Take my purse, just don’t hurt me please!'[SEP]"
    data_ss_peabody['Question'] = "Why did Mrs Peabody say that?"

    data_ss_prisoner = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SS_Prisoner_Text.xlsx")
    data_ss_prisoner[
        'Passage'] = "During the war, the Red army captures a member of the Blue army. They want him to tell them where his army’s tanks are; they know they are either by the sea or in the mountains. They know that the prisoner will not want to tell them, he will want to save his army and so he will certainly lie to them. The prisoner is very brave and very clever, he will not let them find his tanks. The tanks are really in the mountains. Now when the other side asks him where his tanks are, he says, ‘They are in the mountains.'[SEP]"
    data_ss_prisoner['Question'] = "Why did the prisoner say that?"

    data_ss_simon = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SS_Simon_Text.xlsx")
    data_ss_simon[
        'Passage'] = "Simon is a big liar. Simon’s brother Jim knows this, he knows that Simon never tells the truth! Now yesterday Simon stole Jim’s table-tennis paddle, and Jim knows Simon has hidden it somewhere, though he can’t find it. He’s very cross. So he finds Simon and he says, ‘Where is my table-tennis paddle? You must have hidden it either in the cupboard or under your bed, because I’ve looked everywhere else. Where is it, in the cupboard or under your bed?’ Simon tells him the paddle is under his bed.[SEP]"
    data_ss_simon['Question'] = "Why will Jim look in the cupboard for the paddle?"

    data_all_raw = pd.concat([data_ss_brain,
                              data_ss_burglar,
                              data_ss_peabody,
                              data_ss_prisoner,
                              data_ss_simon], ignore_index=True)

    dataset = pd.DataFrame(columns=['Question', 'Answer', 'Score'])

    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']

    return dataset

def shuffle_dataset(dataset):
    index = [i for i in range(len(dataset))]
    np.random.shuffle(index)
    dataset = dataset[index]
    return dataset

def split_train(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

def get_videos():
    frame1 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    imgs_tensor1 = transform_frames(frame1)

    frame2 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    imgs_tensor2 = transform_frames(frame2)

    frame3 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip2.mp4")
    imgs_tensor3 = transform_frames(frame3)

    frame4 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip3.mp4")
    imgs_tensor4 = transform_frames(frame4)

    frame5 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip4.mp4")
    imgs_tensor5 = transform_frames(frame5)

    frame6 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip5.mp4")
    imgs_tensor6 = transform_frames(frame6)

    frames = [imgs_tensor1, imgs_tensor2, imgs_tensor3, imgs_tensor4, imgs_tensor5, imgs_tensor6]

    return frames

def get_newvideos():
    frame1 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    imgs_tensor1 = transform_newframes1(frame1)

    frame2 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    imgs_tensor2 = transform_newframes1(frame2)

    frame3 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip2.mp4")
    imgs_tensor3 = transform_newframes1(frame3)

    frame4 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip3.mp4")
    imgs_tensor4 = transform_newframes1(frame4)

    frame5 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip4.mp4")
    imgs_tensor5 = transform_newframes1(frame5)

    frame6 = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip5.mp4")
    imgs_tensor6 = transform_newframes1(frame6)

    frames = [imgs_tensor1, imgs_tensor2, imgs_tensor3, imgs_tensor4, imgs_tensor5, imgs_tensor6]

    return frames

def text_preprocessing(text):

    text = re.sub(r'(@.*?)[\s]', ' ', text)

    # Replace '&amp;' with '&'
    text = re.sub(r'&amp;', '&', text)

    # Remove trailing whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text