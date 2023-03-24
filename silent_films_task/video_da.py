from codecs import ignore_errors
import pandas as pd
import numpy as np
from video_extract import *
from data_process import *

#setup_seed(512)
#pd.set_option('mode.chained_assignment', None)

def get_data2(frames):
    data_sf1 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_1_Text.xlsx")
    data_sf1['videoclass'] = 1
    data_sf1['Question'] = "Why do you think the men hide?"
    data_sf1['Frames'] = [frames[0]] * len(data_sf1)

    data_sf2 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_2_Text.xlsx")
    data_sf2['videoclass'] = 2
    data_sf2['Question'] = "What do you think the woman is thinking?"
    data_sf2['Frames'] = [frames[1]] * len(data_sf2)

    data_sf3 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_3_Text.xlsx")
    data_sf3['videoclass'] = 3
    data_sf3['Question'] = "Why do you think the driver locks Harold in the van?"
    data_sf3['Frames'] = [frames[2]] * len(data_sf3)

    data_sf4 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_4_Text.xlsx")
    data_sf4['videoclass'] = 4
    data_sf4['Question'] = "What do you think the delivery man is feeling and why?"
    data_sf4['Frames'] = [frames[3]] * len(data_sf4)

    data_sf5 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_5_Text.xlsx")
    data_sf5['videoclass'] = 5
    data_sf5['Question'] = "Why do you think Harold picks up the cat?"
    data_sf5['Frames'] = [frames[4]] * len(data_sf5)

    data_sf6 = pd.read_excel("/data/yanyuliang/IJCAI2023/Data/relabel/SFQuestion_6_Text.xlsx")
    data_sf6['videoclass'] = 6
    data_sf6['Question'] = "Why do you think Harold fans Mildred?"
    data_sf6['Frames'] = [frames[5]] * len(data_sf6)

    data_all_raw = pd.concat([data_sf1, data_sf2, data_sf3, data_sf4, data_sf5, data_sf6], ignore_index=True)

    dataset = pd.DataFrame(columns=['Frames', 'Question', 'Answer', 'Score'])

    dataset['Frames'] = data_all_raw['Frames']
    dataset['Question'] = data_all_raw['Question']
    dataset['Answer'] = data_all_raw['Answer']
    dataset['Score'] = data_all_raw['Score']
    dataset['videoclass'] = data_all_raw['videoclass']

    return dataset

def df_trainsetda(base_train):
    frames = {}
    frames[0] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    frames[1] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    frames[2] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip2.mp4")
    frames[3] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip3.mp4")
    frames[4] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip4.mp4")
    frames[5] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip5.mp4")
    newdataset = base_train
    newdata_num = 0
    for index,row in base_train.iterrows():
        da_prob = np.random.rand()
        if da_prob>0.5:  # 50% augmente the data 
            newdata_num = newdata_num + 1  #count the number of newdata 
            v_class = row['videoclass']
            new_answer = row['Answer']
            newdata = pd.DataFrame({'Question':[row['Question']],'Answer':[new_answer],'Score':[row['Score']],'videoclass':[row['videoclass']],'Frames':[transform_newframes1(frames[v_class-1])]})  # augmente the frames and concat to the new data
            newdataset = pd.concat([newdataset,newdata],ignore_index= True)
            #print((len(newdataset)))
    print("newdatanumber:" + str(newdata_num))  # number of new data
    return newdataset

def df_trainsetdavideo(base_train):  #only augmente the video
    frames = {}
    frames[0] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    frames[1] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip1.mp4")
    frames[2] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip2.mp4")
    frames[3] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip3.mp4")
    frames[4] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip4.mp4")
    frames[5] = get_frames("/data/yanyuliang/IJCAI2023/Data/video_raw/Clip5.mp4")
    newdataset = base_train
    newdata_num = 0
    for index,row in base_train.iterrows():
        da_prob = np.random.rand()
        if da_prob>0.1:  # 50% augmente the data 
            newdata_num = newdata_num+1  #count the number of newdata 
            v_class = row['videoclass']
            #new_answer = textda(row['Answer'])   #augmente the Answer of the data
            newdata = pd.DataFrame({'Question':[row['Question']],'Answer':[row['Answer']],'Score':[row['Score']],'videoclass':[row['videoclass']],'Frames':[transform_newframes1(frames[v_class-1])]})  # augmente the frames and concat to the new data
            newdataset = pd.concat([newdataset,newdata],ignore_index= True)
            #print((len(newdataset)))
    #print("newdatanumber:"+str(newdata_num))  # number of new data
    return newdataset

# frames = get_videos()
# dataset = get_data2(frames)
# df_train_0, df_test = split_train(dataset, 0.1)
# df_train_0 = df_trainsetda(df_train_0)
# df_train_0.to_csv("dataset_afterda.csv",index=False)

# df_train_0 = pd.read_csv("dataset_afterda.csv")
# print(df_train_0)

# data_sf1 = pd.read_excel("/home/yyl/code/Labs/Data/relabel/SFQuestion_1_Text.xlsx")
# data_sf1['videoclass'] = 1
# data_sf1['Question'] = "Why do you think the men hide?"
# data_sf1['Frames'] = [frames[0]] * len(data_sf1)
# testnewdataset = pd.DataFrame(columns=['Frames', 'Question', 'Answer', 'Score', 'videoclass'])
# testnewdataset['Frames'] = data_sf1['Frames']
# testnewdataset['Question'] =data_sf1['Question']
# testnewdataset['Answer'] =data_sf1['Answer']
# testnewdataset['Score'] = data_sf1['Score']
# testnewdataset['videoclass'] = data_sf1['videoclass']
# test = df_trainsetda(testnewdataset)
# print(test)
# print(type(test.Question))
# print(type(test.Answer))
# test2 = test.Question.values + test.Answer.values