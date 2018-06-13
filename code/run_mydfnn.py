import numpy as np
import pandas as pd
import gc
import torch
from myDeepFMM import DeepFM
import h5py


def load_category_index(file_path,field_size):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(field_size):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict


one_hot_features = ['age','gender','education','consumptionAbility','LBS','house','carrier','aid','advertiserId','campaignId','creativeSize','adCategoryId','productType','productId',
                    'age_aid', 'gender_aid', 'LBS_aid', 'education_aid', 'LBS_creativeSize']
text_features = ['os','interest1','interest2','interest3','interest4','interest5','kw1','kw2','kw3','topic1','topic2','topic3','marriageStatus','ct']

field_size = len(one_hot_features+text_features)
one_field = len(one_hot_features)
list_field = len(text_features)
max_num = 86
cate_dict = load_category_index('../data/emb_2.csv',field_size)
embedding_size = 8
list_features = 0
one_features  = 0
for j, f in enumerate(one_hot_features):
    one_features += len(cate_dict[j])
for j, v in enumerate(text_features):
    tj = j + len(one_hot_features)
    list_features += len(cate_dict[tj])




with torch.cuda.device(0),h5py.File('../data/dataset.h5', "r") as dataset:
    deepfm = DeepFM(field_size,one_features,list_features,one_field,list_field,max_num,embedding_size,batch_size = 2048,verbose=True,use_cuda=True, weight_decay=0.00001,use_fm=False,use_ffm=True,use_deep=True).cuda()
    deepfm.fit(dataset,is_valid = True,ealry_stopping=True,refit=False,save_path='../data/ffm.model_1')
    res = pd.read_csv('../data/test2.csv')
    res['score'] = deepfm.predict_proba(dataset)
    res['score'] = res['score'].apply(lambda x: float('%.6f' % x))
    res.to_csv('../data/submission_1.csv',index=False)

