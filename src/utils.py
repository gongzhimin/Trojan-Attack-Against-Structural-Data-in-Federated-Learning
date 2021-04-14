#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
import numpy as np
import pandas as pd
from sampling import sample_iid, sample_noniid

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import SparseFeat, DenseFeat

from collections import Counter


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    data_dir = './data/new_data.csv'
    data = pd.read_csv(data_dir)
    sparse_features = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
                       'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class',
                       'city', 'city_rank', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
                       'up_membership_grade', 'consume_purchase', 'indu_name', 'pt_d']

    dense_features = ['age', 'device_size', 'his_app_size', 'his_on_shelf_time', 'app_score', 'list_time',
                      'device_price', 'up_life_duration', 'up_membership_grade', 'membership_life_duration',
                      'consume_purchase', 'communication_avgonline_30d']

    # data[sparse_features] = data[sparse_features].fillna('-1', )
    # data[dense_features] = data[dense_features].fillna(0, )


    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])


    # 2.generate input data for model
    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}
    # train_model_input = {name: train[name] for name in feature_names}
    # test_model_input = {name: test[name] for name in feature_names}


    # 2.1 sample training data amongst users
    if args['iid']:
        # Sample IID user data from Mnist
        user_groups = sample_iid(train, args['num_users'])
    else:
        # Sample Non-IID user data from Mnist
        if args["unequal"]:
            # Chose uneuqal splits for every user
            # user_groups = mnist_noniid_unequal(train, args.num_users)
            raise NotImplementedError()
        else:
            # Chose euqal splits for every user
            user_groups = sample_noniid(train, args['num_users'])

    # 3.count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                            for feat in dense_features]


    return (train, test, user_groups),fixlen_feature_columns


def average_weights(w): #, flags, alpha=10):
    """
    Returns the average of the weights.
    """
    # rates = np.array([alpha if x else 1 for x in flags])
    # rates = rates / np.sum(rates)
    # print("rates",rates)
    w_avg = copy.deepcopy(w[0]) # copy the structure from list
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key] #* rates[i]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f"    Model     : {args['model']}")
    # print(f'    Optimizer : {args.optimizer}')
    print(f"    Learning  : {args['lr']}")
    print(f"    Global Rounds   : {args['epochs']}\n")

    print("    Federated parameters:")
    if args['iid']:
        print("    IID")
    else:
        print("    Non-IID")
    print(f"    Fraction of users  : {args['frac']}")
    print(f"    Local Batch size   : {args['local_bs']}")
    print(f"    Local Epochs       : {args['local_ep']}\n")
    return
