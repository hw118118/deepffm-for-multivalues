import h5py
import numpy as np
import pandas as pd
import gc
from time import time


def load_category_index(file_path,field_size):
    f = open(file_path,'r')
    cate_dict = []
    for i in range(field_size):
        cate_dict.append({})
    for line in f:
        datas = line.strip().split(',')
        cate_dict[int(datas[0])][datas[1]] = int(datas[2])
    return cate_dict



def create_dataset(file_path,trainset,validation,testset,one_hot_features,text_features,field_size,one_field,list_field,cate_dict,max_num):
    with h5py.File(file_path,'w') as f:
        ftr = f.create_group('train')
        fv = f.create_group('val')
        fte = f.create_group('test')


        ftr.create_dataset('Xi_one',(trainset.shape[0],one_field),np.int16)
        ftr.create_dataset('Xi_mul', (trainset.shape[0],list_field,max_num),np.int32)
        ftr.create_dataset('y', (trainset.shape[0],),np.int8)

        fv.create_dataset('Xi_one', (validation.shape[0], one_field), np.int16)
        fv.create_dataset('Xi_mul', (validation.shape[0], list_field, max_num), np.int32)
        fv.create_dataset('y', (validation.shape[0],), np.int8)

        fte.create_dataset('Xi_one', (testset.shape[0], one_field), np.int16)
        fte.create_dataset('Xi_mul', (testset.shape[0], list_field, max_num), np.int32)

        print('trainset start...')
        #trainset
        nrows = trainset.shape[0]
        for i in range(0, nrows):
            if (i % 100000 == 0):
                print('row:', i)
            datarow = trainset.loc[i].to_dict()
            y = datarow['label']
            indexs_one = []
            indexs_mul = []
            current_ind = 0
            for j, f in enumerate(one_hot_features):
                if str(datarow[f]) in cate_dict[j]:
                    indices = cate_dict[j][str(datarow[f])] - 1
                    indices += current_ind
                    current_ind += len(cate_dict[j])
                    indexs_one.append(indices)

            current_ind = 0
            for j, v in enumerate(text_features):
                tj = j + len(one_hot_features)
                indices = np.zeros((max_num))
                k = 0
                for f in datarow[v].strip().split(' '):
                    if (f in cate_dict[tj]):
                        indices[k] = cate_dict[tj][f] + current_ind
                        k = k + 1
                indexs_mul.append(indices)
                current_ind += len(cate_dict[tj])
            # start_time = time()
            ftr['Xi_one'][i,...] = np.array(indexs_one)
            ftr['Xi_mul'][i,...] = np.array(indexs_mul)
            ftr['y'][i] = y
            # print('time:',time()-start_time)
        del trainset
        gc.collect()
        print('trainset done!!!')


        print('validation start...')
        # validation
        nrows = validation.shape[0]
        for i in range(0, nrows):
            if (i % 100000 == 0):
                print('row:', i)
            datarow = validation.loc[i].to_dict()
            y = datarow['label']
            indexs_one = []
            indexs_mul = []
            current_ind = 0
            for j, f in enumerate(one_hot_features):
                if str(datarow[f]) in cate_dict[j]:
                    indices = cate_dict[j][str(datarow[f])] - 1
                    indices += current_ind
                    current_ind += len(cate_dict[j])
                    indexs_one.append(indices)

            current_ind = 0
            for j, v in enumerate(text_features):
                tj = j + len(one_hot_features)
                indices = np.zeros((max_num))
                k = 0
                for f in datarow[v].strip().split(' '):
                    if (f in cate_dict[tj]):
                        indices[k] = cate_dict[tj][f] + current_ind
                        k = k + 1
                indexs_mul.append(indices)
                current_ind += len(cate_dict[tj])
            fv['Xi_one'][i, ...] = np.array(indexs_one)
            fv['Xi_mul'][i, ...] = np.array(indexs_mul)
            fv['y'][i] = y
        del validation
        gc.collect()
        print('validation done!!!')


        print('testset start...')
        # testset
        nrows = testset.shape[0]
        for i in range(0, nrows):
            if (i % 100000 == 0):
                print('row:', i)
            datarow = testset.loc[i].to_dict()
            indexs_one = []
            indexs_mul = []
            current_ind = 0
            for j, f in enumerate(one_hot_features):
                if str(datarow[f]) in cate_dict[j]:
                    indices = cate_dict[j][str(datarow[f])] - 1
                    indices += current_ind
                    current_ind += len(cate_dict[j])
                    indexs_one.append(indices)

            current_ind = 0
            for j, v in enumerate(text_features):
                tj = j + len(one_hot_features)
                indices = np.zeros((max_num))
                k = 0
                for f in datarow[v].strip().split(' '):
                    if (f in cate_dict[tj]):
                        indices[k] = cate_dict[tj][f] + current_ind
                        k = k + 1
                indexs_mul.append(indices)
                current_ind += len(cate_dict[tj])
            fte['Xi_one'][i, ...] = np.array(indexs_one)
            fte['Xi_mul'][i, ...] = np.array(indexs_mul)
        del testset
        gc.collect()
        print('testset done!!!')


if __name__ =='__main__':
    trainset = pd.read_csv('../data/trainset_2.csv')
    print(trainset.shape)
    validation = pd.read_csv('../data/validation_2.csv')
    print(validation.shape)
    testset = pd.read_csv('../data/testset_2.csv')
    print(testset.shape)
    one_hot_features = ['age', 'gender', 'education', 'consumptionAbility', 'LBS', 'house', 'carrier', 'aid',
                        'advertiserId', 'campaignId', 'creativeSize', 'adCategoryId', 'productType', 'productId',
                        'age_aid','gender_aid','LBS_aid','education_aid','LBS_creativeSize']
    text_features = ['os','interest1', 'interest2', 'interest3', 'interest4', 'interest5', 'kw1', 'kw2', 'kw3', 'topic1',
                     'topic2', 'topic3', 'marriageStatus', 'ct', 'appIdAction', 'appIdInstall']
    # mix_features = [['age', 'aid'], ['gender', 'aid'], ['LBS', 'aid'], ['education', 'aid'], ['LBS', 'creativeSize']]
    field_size = len(one_hot_features + text_features)
    one_field = len(one_hot_features)
    list_field = len(text_features)
    cate_dict = load_category_index('../data/emb_2.csv', field_size)
    max_num = 86
    create_dataset('../data/dataset.h5',trainset,validation,testset,one_hot_features,text_features,field_size,one_field,list_field,cate_dict,max_num)
