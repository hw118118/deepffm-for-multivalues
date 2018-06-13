
# -*- coding:utf-8 -*-

"""
Created on Dec 10, 2017
@author: jachin,Nie
A pytorch implementation of deepfm
Reference:
[1] DeepFM: A Factorization-Machine based Neural Network for CTR Prediction,
    Huifeng Guo, Ruiming Tang, Yunming Yey, Zhenguo Li, Xiuqiang He.
"""

import os
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_auc_score
from time import time

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.backends.cudnn

"""
    网络结构部分
"""


class DeepFM(torch.nn.Module):
    """
    :parameter
    -------------
    field_size: size of the feature fields
    feature_sizes: a field_size-dim array, sizes of the feature dictionary
    embedding_size: size of the feature embedding
    is_shallow_dropout: bool, shallow part(fm or ffm part) uses dropout or not?
    dropout_shallow: an array of the size of 2, example:[0.5,0.5], the first element is for the-first order part and the second element is for the second-order part
    h_depth: deep network's hidden layers' depth
    deep_layers: a h_depth-dim array, each element is the size of corresponding hidden layers. example:[32,32] h_depth = 2
    is_deep_dropout: bool, deep part uses dropout or not?
    dropout_deep: an array of dropout factors,example:[0.5,0.5,0.5] h_depth=2
    deep_layers_activation: relu or sigmoid etc
    n_epochs: epochs
    batch_size: batch_size
    learning_rate: learning_rate
    optimizer_type: optimizer_type, 'adam', 'rmsp', 'sgd', 'adag'
    is_batch_norm：bool,  use batch_norm or not ?
    verbose: verbose
    weight_decay: weight decay (L2 penalty)
    random_seed: random_seed=950104 someone's birthday, my lukcy number
    use_fm: bool
    use_ffm: bool
    use_deep: bool
    loss_type: "logloss", only
    eval_metric: roc_auc_score
    use_cuda: bool use gpu or cpu?
    n_class: number of classes. is bounded to 1
    greater_is_better: bool. Is the greater eval better?
    Attention: only support logsitcs regression
    """

    def __init__(self, field_size, one_features,list_fetures,one_field,list_field,max_num, embedding_size=4, is_shallow_dropout=False, dropout_shallow=[0.5, 0.5],
                 h_depth=2, deep_layers=[128, 128], is_deep_dropout=True, dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation='relu', n_epochs=64, batch_size=256, learning_rate=0.001,
                 optimizer_type='adam', is_batch_norm=True, verbose=False, random_seed=2018, weight_decay=0.0,
                 use_fm=True, use_ffm=False, use_deep=True, loss_type='logloss', eval_metric=roc_auc_score,
                 use_cuda=True, n_class=1, greater_is_better=True
                 ):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        # self.feature_sizes = feature_sizes
        self.one_features = one_features
        self.list_fetures = list_fetures
        self.one_field = one_field
        self.list_field = list_field
        self.max_num = max_num
        self.embedding_size = embedding_size
        self.is_shallow_dropout = is_shallow_dropout
        self.dropout_shallow = dropout_shallow
        self.h_depth = h_depth
        self.deep_layers = deep_layers
        self.is_deep_dropout = is_deep_dropout
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_batch_norm = is_batch_norm
        self.verbose = verbose
        self.weight_decay = weight_decay
        self.random_seed = random_seed
        self.use_fm = use_fm
        self.use_ffm = use_ffm
        self.use_deep = use_deep
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.use_cuda = use_cuda
        self.n_class = n_class
        self.greater_is_better = greater_is_better

        torch.manual_seed(self.random_seed)

        """
            check cuda
        """
        if self.use_cuda and not torch.cuda.is_available():
            self.use_cuda = False
            print("Cuda is not available, automatically changed into cpu model")

        """
            check use fm or ffm
        """
        if self.use_fm and self.use_ffm:
            print("only support one type only, please make sure to choose only fm or ffm part")
            exit(1)
        elif self.use_fm and self.use_deep:
            print("The model is deepfm(fm+deep layers)")
        elif self.use_ffm and self.use_deep:
            print("The model is deepffm(ffm+deep layers)")
        elif self.use_fm:
            print("The model is fm only")
        elif self.use_ffm:
            print("The model is ffm only")
        elif self.use_deep:
            print("The model is deep layers only")
        else:
            print("You have to choose more than one of (fm, ffm, deep) models to use")
            exit(1)

        """
            bias
        """
        if self.use_fm or self.use_ffm:
            self.bias = torch.nn.Parameter(torch.randn(1))
        """
            fm part
        """
        if self.use_fm:
            print("Init fm part")
            self.fm_first_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            self.fm_second_order_embeddings = nn.ModuleList(
                [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])
            if self.dropout_shallow:
                self.fm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init fm part succeed")

        """
            ffm part
        """
        if self.use_ffm:
            print("Init ffm part")
            # self.ffm_first_order_embeddings = nn.ModuleList(
            #     [nn.Embedding(feature_size, 1) for feature_size in self.feature_sizes])
            # total_features one-hot
            self.ffm_first_order_embeddings_1 = nn.Embedding(self.one_features,1)  #totalsize*1
            # list features multi_values
            self.ffm_first_order_embeddings_2 = nn.Embedding(self.list_fetures+1,1,0) #list_fetures*1
            # self.ffm_first_order_embeddings_2_pool = nn.AvgPool1d(self.max_num) #list_fetures*1
            if self.dropout_shallow:
                self.ffm_first_order_dropout = nn.Dropout(self.dropout_shallow[0])
            # self.ffm_second_order_embeddings = nn.ModuleList(
            #     [nn.ModuleList([nn.Embedding(feature_size, self.embedding_size) for i in range(self.field_size)]) for
            #      feature_size in self.feature_sizes])
            self.ffm_second_order_embeddings_1 = nn.ModuleList([nn.Embedding(self.one_features,self.embedding_size) for i in range(self.field_size)])
            self.ffm_second_order_embeddings_2 = nn.ModuleList([nn.Embedding(self.list_fetures+1,self.embedding_size,0) for i in range(self.field_size)])
            # self.ffm_second_order_embeddings_2_pool = nn.AvgPool2d((self.max_num,1))
            if self.dropout_shallow:
                self.ffm_second_order_dropout = nn.Dropout(self.dropout_shallow[1])
            print("Init ffm part succeed")

        """
            deep part
        """
        if self.use_deep:
            print("Init deep part")
            if not self.use_fm and not self.use_ffm:
                self.fm_second_order_embeddings = nn.ModuleList(
                    [nn.Embedding(feature_size, self.embedding_size) for feature_size in self.feature_sizes])

            if self.is_deep_dropout:
                self.linear_0_dropout = nn.Dropout(self.dropout_deep[0])

            self.linear_1 = nn.Linear(self.field_size * (self.field_size - 1) // 2*self.embedding_size, deep_layers[0])
            if self.is_batch_norm:
                self.batch_norm_1 = nn.BatchNorm1d(deep_layers[0])
            if self.is_deep_dropout:
                self.linear_1_dropout = nn.Dropout(self.dropout_deep[1])
            for i, h in enumerate(self.deep_layers[1:], 1):
                setattr(self, 'linear_' + str(i + 1), nn.Linear(self.deep_layers[i - 1], self.deep_layers[i]))
                if self.is_batch_norm:
                    setattr(self, 'batch_norm_' + str(i + 1), nn.BatchNorm1d(deep_layers[i]))
                if self.is_deep_dropout:
                    setattr(self, 'linear_' + str(i + 1) + '_dropout', nn.Dropout(self.dropout_deep[i + 1]))

            print("Init deep part succeed")

        print("Init succeed")

    def avg_dym(self,dynamic_sum_embedding,len):
        dynamic_lengths_tensor = torch.autograd.Variable(len.data.float())
        dynamic_lengths_tensor = dynamic_lengths_tensor.view(-1, 1).expand_as(dynamic_sum_embedding)
        dynamic_avg_embedding = dynamic_sum_embedding / dynamic_lengths_tensor
        dynamic_embeddings = dynamic_avg_embedding.view(-1, self.list_field, self.embedding_size)
        return dynamic_embeddings


    def forward(self, Xi_one, Xi_mul,Xi_mle):
        """
        :param Xi_one: index input tensor that has one values, batch_size * k_one
        :param Xi_mul: index input tensor that has multiple values, batch_size * k_mul * m
        :return: the last output
        """
        """
            fm part
        """
        # if self.use_fm:
        #     fm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                               enumerate(self.fm_first_order_embeddings)]
        #     fm_first_order = torch.cat(fm_first_order_emb_arr, 1)
        #     if self.is_shallow_dropout:
        #         fm_first_order = self.fm_first_order_dropout(fm_first_order)
        #
        #     # use 2xy = (x+y)^2 - x^2 - y^2 reduce calculation
        #     fm_second_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
        #                                enumerate(self.fm_second_order_embeddings)]
        #     fm_sum_second_order_emb = sum(fm_second_order_emb_arr)
        #     fm_sum_second_order_emb_square = fm_sum_second_order_emb * fm_sum_second_order_emb  # (x+y)^2
        #     fm_second_order_emb_square = [item * item for item in fm_second_order_emb_arr]
        #     fm_second_order_emb_square_sum = sum(fm_second_order_emb_square)  # x^2+y^2
        #     fm_second_order = (fm_sum_second_order_emb_square - fm_second_order_emb_square_sum) * 0.5
        #     if self.is_shallow_dropout:
        #         fm_second_order = self.fm_second_order_dropout(fm_second_order)

        """
            ffm part
        """
        if self.use_ffm:
            # ffm_first_order_emb_arr = [(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
            #                            enumerate(self.ffm_first_order_embeddings)]
            #Xi:None*one_filed_size
            ffm_first_order_emb_1 = torch.sum(self.ffm_first_order_embeddings_1(Xi_one),2)#None*F_one
            ffm_first_order_emb_2 = torch.sum(torch.sum(self.ffm_first_order_embeddings_2(Xi_mul),3),2).view(-1,1)#None*F_mul*m
            dynamic_lengths_tensor = torch.autograd.Variable(Xi_mle.data.float())
            dynamic_lengths_tensor = dynamic_lengths_tensor.view(-1, 1)
            ffm_first_order_emb_2 = ffm_first_order_emb_2 / dynamic_lengths_tensor
            ffm_first_order_emb_2 = ffm_first_order_emb_2.view(-1, self.list_field)
            # print(Xi_one.shape)
            # print(ffm_first_order_emb_1.shape)
            # print(ffm_first_order_emb_2.shape)
            # ffm_first_order_emb_2 = torch.sum(self.ffm_first_order_embeddings_2_pool(ffm_first_order_emb_2),2)#None*k_m
            ffm_first_order = torch.cat((ffm_first_order_emb_1,ffm_first_order_emb_2), 1)
            if self.is_shallow_dropout:
                ffm_first_order = self.ffm_first_order_dropout(ffm_first_order)
            # ffm_second_order_emb_arr = [[(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for emb in f_embs] for
            #                             i, f_embs in enumerate(self.ffm_second_order_embeddings)]
            ffm_second_order_emb_1 = [emb(Xi_one) for i,emb in enumerate(self.ffm_second_order_embeddings_1)]
            # ffm_second_order_emb_2 = [torch.sum(self.ffm_second_order_embeddings_2_pool(emb(Xi_mul)),2) for i, emb in enumerate(self.ffm_second_order_embeddings_2)]
            ffm_second_order_emb_2 = [self.avg_dym(torch.sum(emb(Xi_mul),2).view(-1,self.embedding_size),Xi_mle) for i, emb in enumerate(self.ffm_second_order_embeddings_2)]
            # print(ffm_second_order_emb_1[0].shape)
            # print(ffm_second_order_emb_2[0].shape)
            # print(ffm_second_order_emb_2[0].shape)
            ffm_second_order_emb_1 = torch.stack(ffm_second_order_emb_1, 2)
            ffm_second_order_emb_2 = torch.stack(ffm_second_order_emb_2, 2)
            # print(ffm_second_order_emb_2.shape)
            # ffm_second_order_emb_arr = ffm_second_order_emb_1 + ffm_second_order_emb_2
            # print(len(ffm_second_order_emb_arr))
            ffm_second_order_emb_arr = torch.cat((ffm_second_order_emb_1,ffm_second_order_emb_2),1)
            # print(ffm_second_order_emb_arr.shape)
            ffm_second_order_emb_t = torch.transpose(ffm_second_order_emb_arr,1,2)
            ffm_second_order = ffm_second_order_emb_arr*ffm_second_order_emb_t
            ffm_second_order = torch.transpose(ffm_second_order,1,3)
            lower_mask_op = torch.ones(self.field_size,self.field_size)
            lower_mask_op = torch.tril(lower_mask_op,-1)
            lower_mask_op = lower_mask_op.expand_as(ffm_second_order)
            lower_mask = torch.ge(lower_mask_op,1)
            ffm_second_order = torch.masked_select(ffm_second_order,lower_mask.cuda())
            ffm_second_order = ffm_second_order.view(-1, self.embedding_size,self.field_size * (self.field_size-1) // 2)
            ffm_second_order = torch.transpose(ffm_second_order,1,2)
            # print(ffm_second_order.shape)
            # print(self.field_size * (self.field_size - 1) // 2*self.embedding_size)
            ffm_second_order = ffm_second_order.reshape(-1, self.field_size * (self.field_size - 1) // 2*self.embedding_size)
            # ffm_second_order = torch.sum(ffm_second_order,1)
            # ffm_second_order = 0.5*torch.sum(ffm_second_order,1)
            # print(ffm_second_order.shape)
            # ffm_wij_arr = []
            # for i in range(self.field_size):
            #     for j in range(i + 1, self.field_size):
            #         ffm_wij_arr.append(ffm_second_order_emb_arr[i][j] * ffm_second_order_emb_arr[j][i])
            # ffm_second_order = sum(ffm_wij_arr)
            if self.is_shallow_dropout:
                ffm_second_order = self.ffm_second_order_dropout(ffm_second_order)

        """
            deep part
        """
        if self.use_deep:
            if self.use_fm:
                print('please use_fmm')
                # deep_emb = torch.cat(fm_second_order_emb_arr, 1)
            elif self.use_ffm:
                # deep_emb = torch.cat([sum(ffm_second_order_embs) for ffm_second_order_embs in ffm_second_order_emb_arr],
                #                      1)
                # deep_emb = torch.sum(ffm_second_order_emb_arr,2)
                # deep_emb = deep_emb.view(-1,self.field_size*self.embedding_size)
                deep_emb = ffm_second_order
                # deep_emb = torch.reshape(deep_emb,(-1,self.field_size*self.embedding_size))
                # print(deep_emb.shape)
            else:
                print('please use_fmm')
                # deep_emb = torch.cat([(torch.sum(emb(Xi[:, i, :]), 1).t() * Xv[:, i]).t() for i, emb in
                #                       enumerate(self.fm_second_order_embeddings)], 1)

            if self.deep_layers_activation == 'sigmoid':
                activation = F.sigmoid
            elif self.deep_layers_activation == 'tanh':
                activation = F.tanh
            else:
                activation = F.relu
            if self.is_deep_dropout:
                deep_emb = self.linear_0_dropout(deep_emb)
            x_deep = self.linear_1(deep_emb)
            if self.is_batch_norm:
                x_deep = self.batch_norm_1(x_deep)
            x_deep = activation(x_deep)
            if self.is_deep_dropout:
                x_deep = self.linear_1_dropout(x_deep)
            for i in range(1, len(self.deep_layers)):
                x_deep = getattr(self, 'linear_' + str(i + 1))(x_deep)
                if self.is_batch_norm:
                    x_deep = getattr(self, 'batch_norm_' + str(i + 1))(x_deep)
                x_deep = activation(x_deep)
                if self.is_deep_dropout:
                    x_deep = getattr(self, 'linear_' + str(i + 1) + '_dropout')(x_deep)
        """
            sum
        """
        if self.use_fm and self.use_deep:
            print('please use_fmm')
            # total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + torch.sum(x_deep, 1) + self.bias
        elif self.use_ffm and self.use_deep:
            total_sum = torch.sum(ffm_first_order, 1) + torch.sum(ffm_second_order, 1) + torch.sum(x_deep,1) + self.bias
        elif self.use_fm:
            print('please use_fmm')
            # total_sum = torch.sum(fm_first_order, 1) + torch.sum(fm_second_order, 1) + self.bias
        elif self.use_ffm:
            total_sum = torch.sum(ffm_first_order, 1) + torch.sum(ffm_second_order, 1) + self.bias
        else:
            total_sum = torch.sum(x_deep, 1)
        return total_sum

    def get_batches(self,dataset,group):
        if(group=='train' or group=='val'):
            data_group = dataset[group]
            for start in range(0, data_group['Xi_one'].shape[0], self.batch_size):
                end = min(start + self.batch_size, data_group['Xi_one'].shape[0])
                yield data_group['Xi_one'][start:end],data_group['Xi_mul'][start:end,0:14],data_group['Xi_mle'][start:end,0:14].astype(np.int16), data_group['y'][start:end].astype(np.int16)
        elif(group=='test'):
            data_group = dataset[group]
            for start in range(0, data_group['Xi_one'].shape[0], self.batch_size):
                end = min(start + self.batch_size, data_group['Xi_one'].shape[0])
                yield data_group['Xi_one'][start:end], data_group['Xi_mul'][start:end,0:14],data_group['Xi_mle'][start:end,0:14].astype(np.int16)
        else:
            data_group = dataset['train']
            for start in range(0, data_group['Xi_one'].shape[0], self.batch_size):
                end = min(start + self.batch_size, data_group['Xi_one'].shape[0])
                yield data_group['Xi_one'][start:end], data_group['Xi_mul'][start:end,0:14], data_group['Xi_mle'][start:end,0:14].astype(np.int16),data_group['y'][start:end].astype(np.int16)
            data_group = dataset['val']
            for start in range(0, data_group['Xi_one'].shape[0], self.batch_size):
                end = min(start + self.batch_size, data_group['Xi_one'].shape[0])
                yield data_group['Xi_one'][start:end], data_group['Xi_mul'][start:end,0:14], data_group['Xi_mle'][start:end,0:14].astype(np.int16),data_group['y'][start:end].astype(np.int16)


    def fit(self, dataset, is_valid = False,ealry_stopping=False, refit=False, save_path=None):
        """
        :param Xi_train: [[ind1_1, ind1_2, ...], [ind2_1, ind2_2, ...], ..., [indi_1, indi_2, ..., indi_j, ...], ...]
                        indi_j is the feature index of feature field j of sample i in the training set
        :param Xv_train: [[val1_1, val1_2, ...], [val2_1, val2_2, ...], ..., [vali_1, vali_2, ..., vali_j, ...], ...]
                        vali_j is the feature value of feature field j of sample i in the training set
                        vali_j can be either binary (1/0, for binary/categorical features) or float (e.g., 10.24, for numerical features)
        :param y_train: label of each sample in the training set
        :param Xi_valid: list of list of feature indices of each sample in the validation set
        :param Xv_valid: list of list of feature values of each sample in the validation set
        :param y_valid: label of each sample in the validation set
        :param ealry_stopping: perform early stopping or not
        :param refit: refit the model on the train+valid dataset or not
        :param save_path: the path to save the model
        :return:
        """
        """
        pre_process
        """
        if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            print("Save path is not existed!")
            return

        if self.verbose:
            print("pre_process data ing...")
        # Xi_train = np.array(Xi_train).reshape((-1, self.field_size, 1))
        # Xv_train = np.array(Xv_train)
        # Xi_one = np.array(Xi_one).reshape((-1, self.one_field))
        # Xi_mul = np.array(Xi_mul).reshape((-1, self.list_field,self.max_num))
        # y_train = np.array(y_train)
        # x_size = y_train.shape[0]
        # if Xi_one_valid!=None:
        #     Xi_one_valid = np.array(Xi_one_valid).reshape((-1, self.one_field))
        #     Xi_mul_valid = np.array(Xi_mul_valid).reshape((-1, self.list_field,self.max_num))
        #     y_valid = np.array(y_valid)
        #     x_valid_size = y_valid.shape[0]
        #     is_valid = True
        if self.verbose:
            print("pre_process data finished")

        """
            train model
        """
        model = self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        criterion = F.binary_cross_entropy_with_logits

        best_train_score = 0
        valid_result = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            batch_iter = dataset['train']['Xi_one'].shape[0] // self.batch_size
            epoch_begin_time = time()
            batch_begin_time = time()
            train_batches = self.get_batches(dataset,'train')
            for i in range(batch_iter + 1):
                Xi_one_batch, Xi_mul_batch, Xi_ml_batch,y_batch = next(train_batches)
                batch_xi_one = Variable(torch.LongTensor (Xi_one_batch))
                batch_xv_mul = Variable(torch.LongTensor (Xi_mul_batch))
                batch_xv_ml = Variable(torch.LongTensor(Xi_ml_batch))
                # print(batch_xv_ml.cpu().data.numpy())
                batch_y = Variable(torch.FloatTensor(y_batch))
                if self.use_cuda:
                    batch_xio, batch_xim,batch_ximl,batch_y = batch_xi_one.cuda(), batch_xv_mul.cuda(),batch_xv_ml.cuda(), batch_y.cuda()
                # time3 = time()
                optimizer.zero_grad()
                outputs = model(batch_xio, batch_xim,batch_ximl)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                # time4 = time()
                # print('model train time:')
                # print(time4 - time3)
                total_loss += loss.data[0]
                if self.verbose:
                    if i % 100 == 99:  # print every 100 mini-batches
                        eval = self.evaluate(batch_xio, batch_xim, batch_ximl,batch_y)
                        print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                              (epoch + 1, i + 1, total_loss / 100.0, eval, time() - batch_begin_time))
                        total_loss = 0.0
                        batch_begin_time = time()

            # train_loss, train_eval = self.eval_by_batch(Xi_one, Xi_mul, y_train, x_size)
            # train_result.append(train_eval)
            # print('*' * 50)
            # print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
            #       (epoch + 1, train_loss, train_eval, time() - epoch_begin_time))
            # print('*' * 50)

            if is_valid:
                valid_loss, valid_eval = self.eval_by_batch(dataset,'val')
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval, time() - epoch_begin_time))
                print('*' * 50)
            if save_path:
                torch.save(self.state_dict(), save_path)
            if is_valid and ealry_stopping and self.training_termination(valid_result):
                print("early stop at [%d] epoch!" % (epoch + 1))
                # train_loss, train_eval = self.eval_by_batch(dataset,'train')
                # best_train_score = train_eval
                # print('*' * 50)
                # print('best:[%d] loss: %.6f metric: %.6f' %
                #       (epoch + 1, train_loss, best_train_score))
                # print('*' * 50)
                break

        # fit a few more epoch on train+valid until result reaches the best_train_score
        if is_valid and refit:
            if self.verbose:
                print("refitting the model")
            # if self.greater_is_better:
            #     best_epoch = np.argmax(valid_result)
            # else:
            #     best_epoch = np.argmin(valid_result)
            # Xi_train = np.concatenate((Xi_one, Xi_one_valid))
            # Xv_train = np.concatenate((Xi_mul, Xi_mul_valid))
            # y_train = np.concatenate((y_train, y_valid))
            # x_size = dataset['train']['Xi_one'].shape[0] + dataset['val']['Xi_one'].shape[0]
            # self.shuffle_in_unison_scary(Xi_train, Xv_train, y_train)
            batch_iter = (dataset['train']['Xi_one'].shape[0] // self.batch_size)+1
            batch_iter += dataset['val']['Xi_one'].shape[0] // self.batch_size
            for epoch in range(self.n_epochs):
                train_batches = self.get_batches(dataset,'all')
                for i in range(batch_iter + 1):
                    Xi_one_batch, Xi_mul_batch, Xi_ml_batch, y_batch = next(train_batches)
                    batch_xi_one = Variable(torch.LongTensor(Xi_one_batch))
                    batch_xv_mul = Variable(torch.LongTensor(Xi_mul_batch))
                    batch_xv_ml = Variable(torch.LongTensor(Xi_ml_batch))
                    batch_y = Variable(torch.FloatTensor(y_batch))
                    if self.use_cuda:
                        batch_xio, batch_xim, batch_ximl, batch_y = batch_xi_one.cuda(), batch_xv_mul.cuda(), batch_xv_ml.cuda(), batch_y.cuda()
                    optimizer.zero_grad()
                    outputs = model(batch_xio, batch_xim,batch_ximl)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                train_loss, train_eval = self.eval_by_batch(dataset, 'all')
                if save_path:
                    torch.save(self.state_dict(), save_path)
                if abs(best_train_score - train_eval) < 0.001 or \
                        (self.greater_is_better and train_eval > best_train_score) or \
                        ((not self.greater_is_better) and train_eval < best_train_score):
                    break
            if self.verbose:
                print("refit finished")

    def eval_by_batch(self,dataset,group):
        total_loss = 0.0
        y_pred = []
        batch_size = self.batch_size
        if(group!='all'):
            batch_iter = dataset[group]['Xi_one'].shape[0] // batch_size
        else:
            batch_iter = (dataset['train']['Xi_one'].shape[0] // self.batch_size)+1
            batch_iter += dataset['val']['Xi_one'].shape[0] // self.batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        train_batches = self.get_batches(dataset,group)
        for i in range(batch_iter + 1):
            Xi_one_batch, Xi_mul_batch, Xi_ml_batch,y_batch = next(train_batches)
            batch_xi_one = Variable(torch.LongTensor (Xi_one_batch))
            batch_xi_mul = Variable(torch.LongTensor (Xi_mul_batch))
            batch_xv_ml = Variable(torch.LongTensor(Xi_ml_batch))
            batch_y = Variable(torch.FloatTensor(y_batch))
            if self.use_cuda:
                batch_xi_one, batch_xi_mul,batch_xv_ml, batch_y = batch_xi_one.cuda(), batch_xi_mul.cuda(),batch_xv_ml.cuda(), batch_y.cuda()
            outputs = model(batch_xi_one, batch_xi_mul,batch_xv_ml)
            pred = F.sigmoid(outputs).cpu()
            y_pred.extend(pred.data.numpy())
            loss = criterion(outputs, batch_y)
            total_loss += loss.data[0]
        total_metric = self.eval_metric(dataset[group]['y'], y_pred)
        return total_loss / batch_iter, total_metric

    # shuffle three lists simutaneously
    def shuffle_in_unison_scary(self, a, b, c):
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def training_termination(self, valid_result):
        if len(valid_result) > 4:
            if self.greater_is_better:
                if valid_result[-1] < valid_result[-2] and \
                                valid_result[-2] < valid_result[-3] and \
                                valid_result[-3] < valid_result[-4]:
                    return True
            else:
                if valid_result[-1] > valid_result[-2] and \
                                valid_result[-2] > valid_result[-3] and \
                                valid_result[-3] > valid_result[-4]:
                    return True
        return False

    def predict(self, dataset):
        """
        :param Xi: the same as fit function
        :param Xv: the same as fit function
        :return: output, ont-dim array
        """
        train_batches = self.get_batches(dataset, 'test')
        batch_iter = dataset['test']['Xi_one'].shape[0] // self.batch_size
        y_pred = []
        for i in range(batch_iter + 1):
        # Xi_one = np.array(Xi_one).reshape((-1, self.one_field))
        # Xi_mul = np.array(Xi_mul).reshape((-1, self.list_field,self.max_num))
        # Xi_one = Variable(torch.ShortTensor(Xi_one))
        # Xi_mul = Variable(torch.IntTensor(Xi_mul))
            Xi_one_batch, Xi_mul_batch,Xi_ml_batch, y_batch = next(train_batches)
            batch_xi_one = Variable(torch.LongTensor (Xi_one_batch))
            batch_xi_mul = Variable(torch.LongTensor (Xi_mul_batch))
            batch_xv_ml = Variable(torch.LongTensor(Xi_ml_batch))
            if self.use_cuda and torch.cuda.is_available():
                batch_xi_one, batch_xi_mul,batch_xv_ml = batch_xi_one.cuda(), batch_xi_mul.cuda(),batch_xv_ml.cuda()
            model = self.eval()
            pred = F.sigmoid(model(batch_xi_one, batch_xi_mul,batch_xv_ml)).cpu()
            y_pred.extend(pred.data.numpy())

        return (np.array(y_pred) > 0.5)

    def predict_proba(self, dataset):
        # Xi_one = np.array(Xi_one).reshape((-1, self.one_field))
        # Xi_mul = np.array(Xi_mul).reshape((-1, self.list_field, self.max_num))
        # Xi_one = Variable(torch.ShortTensor(Xi_one))
        # Xi_mul = Variable(torch.IntTensor(Xi_mul))
        train_batches = self.get_batches(dataset, 'test')
        batch_iter = dataset['test']['Xi_one'].shape[0] // self.batch_size
        y_pred = []
        for i in range(batch_iter + 1):
            Xi_one_batch, Xi_mul_batch, Xi_ml_batch, y_batch = next(train_batches)
            batch_xi_one = Variable(torch.LongTensor(Xi_one_batch))
            batch_xi_mul = Variable(torch.LongTensor(Xi_mul_batch))
            batch_xv_ml = Variable(torch.LongTensor(Xi_ml_batch))
            if self.use_cuda and torch.cuda.is_available():
                batch_xi_one, batch_xi_mul, batch_xv_ml = batch_xi_one.cuda(), batch_xi_mul.cuda(), batch_xv_ml.cuda()
            model = self.eval()
            pred = F.sigmoid(model(batch_xi_one, batch_xi_mul, batch_xv_ml)).cpu()
            y_pred.extend(pred.data.numpy())
        return y_pred

    def inner_predict(self, Xi_one, Xi_mul,Xi_mle):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi_one, Xi_mul,Xi_mle)).cpu()
        return (pred.data.numpy() > 0.5)

    def inner_predict_proba(self, Xi_one, Xi_mul,Xi_mle):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :return: output, numpy
        """
        model = self.eval()
        pred = F.sigmoid(model(Xi_one, Xi_mul,Xi_mle)).cpu()
        return pred.data.numpy()

    def evaluate(self, Xi_one, Xi_mul,Xi_mle, y):
        """
        :param Xi: tensor of feature index
        :param Xv: tensor of feature value
        :param y: tensor of labels
        :return: metric of the evaluation
        """

        y_pred = self.inner_predict_proba(Xi_one, Xi_mul,Xi_mle)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)

