#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm
import yaml
import argparse

import torch
from tensorboardX import SummaryWriter

# from src.options import args_parser
from update import LocalUpdate
from utils import get_dataset, average_weights, exp_details
from deepctr_torch.inputs import get_feature_names
from sklearn.metrics import log_loss, roc_auc_score

from deepctr_torch.models import DeepFM

if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    params = parser.parse_args()

    with open(f'../{params.params}', 'r') as f:
        args = yaml.load(f)


    exp_details(args)
    exit(0)
    if args['gpu']:
        torch.cuda.set_device(args['gpu'])
    device = 'cuda' if args['gpu'] else 'cpu'

    # load dataset and user groups  # prepare feature for model
    (train_dataset, test_dataset, user_groups),fixlen_feature_columns = get_dataset(args)

    # count #unique features for each sparse field,and record dense feature field name

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)


    # BUILD MODEL
    if args['model'].lower() == 'deepfm':
        # 4.Define Model,train,predict and evaluate
        global_model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
        global_model.compile("adam", "binary_crossentropy",
                             metrics=['binary_crossentropy'], )
    else:
        exit('Error: unrecognized model')

    # # Set the model to train and send it to device.
    # global_model.to(device)
    # global_model.train() # torch claim


    # print(global_model)


    # copy weights
    global_weights = global_model.state_dict()
    # print(global_weights.keys())


    # Training
    # train_loss, train_accuracy = [], []
    # val_acc_list, net_list = [], []
    # cv_loss, cv_acc = [], []
    # print_every = 2
    # val_loss_pre, counter = 0, 0

    # temp test data

    test_model_input = {name: test_dataset[name] for name in feature_names}

    # for comparsion
    # best_model = copy.deepcopy(global_model)
    min_loss = 1000.0
    max_auc = -1.0

    for epoch in tqdm(range(args['epochs'])):
        local_weights= [] #, local_losses , []
        print(f'\n | Global Training Round : {epoch+1} |\n')
        # frac default 0.1; num_users default 100
        m = max(int(args['frac'] * args['num_users']), 1)
        # 100 randomly select 10 as training client
        idxs_users = np.random.choice(range(args['num_users']), m, replace=False)



        for idx in idxs_users: # 10 random users
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)
            w = local_model.update_weights(
                model=copy.deepcopy(global_model), features=feature_names)

            local_weights.append(copy.deepcopy(w))
            # local_losses.append(copy.deepcopy(loss))


        # update global weights
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # temp test
        pred_ans = global_model.predict(test_model_input, batch_size=256)
        logloss = log_loss(test_dataset['label'].values, pred_ans)
        aucscore = roc_auc_score(test_dataset['label'].values, pred_ans)
        print("test LogLoss", round(logloss, 4))
        print("test AUC", round(aucscore, 4))
        if aucscore > max_auc:
            # best_model = copy.deepcopy(global_model)
            min_loss = logloss
            max_auc = aucscore



    print("|---- Min log loss: {:.4f}%".format(min_loss))
    print("|---- Best AUC: {:.4f}%".format(max_auc))
    print("test done")
