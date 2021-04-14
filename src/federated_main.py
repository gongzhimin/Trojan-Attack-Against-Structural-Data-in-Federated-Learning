#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy
import time
import random
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
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
from deepctr_torch.models import DeepFM
import math


def Logloss_loc(y_true, y_pred):
    eps = 1e-15

    # Prepare numpy array data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps to prevent the loss value from becoming `nan`
    p = np.clip(y_pred, eps, 1-eps)
    loss = 0.0
    for i in range(len(y_true)):
        loss += (- y_true[i] * math.log(p[i]) -
                 (1 - y_true[i]) * math.log(1-p[i]))

    return loss / len(y_true)


def poison_data(args, test_dataset, prate=1):

    poi_dataset = copy.deepcopy(test_dataset)
    poison_patterns = args['0_poison_pattern']
    poison_batch = int(len(poi_dataset) * prate)
    namelist = poi_dataset._stat_axis.values.tolist()
    poison_idx = random.sample(namelist, poison_batch)
    poi_dataset.loc[poison_idx, 'label'] = int(1)

    trigger = {'slot_id': 6, 'spread_app_id': 66, 'tags': 27, 'app_first_class': 0,
               'app_score': 1.0, 'list_time': 1.0, 'device_price': 1.0, 'up_life_duration': 1.0}

    for feat in poison_patterns:
        # poi_dataset.loc[poison_idx, feat] = 0  # trigger[feat]
        poi_dataset.loc[poison_idx, feat] = trigger[feat]
    return poi_dataset


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('./logs')

    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    params = parser.parse_args()

    with open(f'./{params.params}', 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    exp_details(args)

    if args['gpu']:
        torch.cuda.set_device(args['gpu'])
        print('gpu')

    device = 'cuda' if args['gpu'] else 'cpu'

    # load dataset and user groups  # prepare feature for model
    (train_dataset, test_dataset, user_groups), fixlen_feature_columns = get_dataset(args)

    # count #unique features for each sparse field,and record dense feature field name
    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    # BUILD MODEL
    if args['model'].lower() == 'deepfm':
        # Define Model,train,predict and evaluate
        global_model = DeepFM(linear_feature_columns,
                              dnn_feature_columns, task='binary')
        global_model.compile("adam", "binary_crossentropy",
                             metrics=['binary_crossentropy'], )
        # global_model = torch.load("../save/cleanmodel.pth")
        global_model.train()
    else:
        exit('Error: unrecognized model')

    # # Set the model to train and send it to device.
    # global_model.to(device)
    # global_model.train() # torch claim

    # copy weights
    global_weights = global_model.state_dict()

    # test data for auc
    test_model_input = {name: test_dataset[name] for name in feature_names}

    pos_idx = np.where(test_dataset['label'] == 1)[0]
    neg_idx = np.delete(
        np.array([i for i in range(len(test_dataset))]), pos_idx)

    # poi test data for attack
    print("poison test data")

    poi_dataset = poison_data(args, test_dataset)
    poi_model_input = {name: poi_dataset[name] for name in feature_names}

    # for comparsion
    min_loss = 1000.0
    max_auc = -1.0

    benign_namelist = list(
        set(list(range(args['num_users']))) - set(args['adversary_list']))

    for epoch in tqdm(range(args['epochs'])):

        print(f'\n | Global Training Round : {epoch} |\n')
        # frac default 0.1; num_users default 100
        m = max(int(args['frac'] * args['num_users']), 1)
        # 100 randomly select 10 as training client
        idxs_users = np.random.choice(
            range(args['num_users']), m, replace=False)

        adversarial_name_keys = []
        for idx in range(0, len(args['adversary_list'])):
            if epoch in args[str(idx) + '_poison_epochs']:
                if args['adversary_list'][idx] not in adversarial_name_keys:
                    adversarial_name_keys.append(args['adversary_list'][idx])
        # if not in there epoch, treat adv as benign
        nonattacker = []
        for adv in args['adversary_list']:
            if adv not in adversarial_name_keys:
                nonattacker.append(copy.deepcopy(adv))
        # benign agent for each epoch, 6
        benign_num = m - len(adversarial_name_keys)
        random_agent_name_keys = random.sample(
            benign_namelist + nonattacker, benign_num)
        idxs_users = adversarial_name_keys + random_agent_name_keys

        local_poison_epochs = []
        local_weights = []
        local_poisoned = []
        for idx in idxs_users:  # 10 agents including 4 advs
            adversarial_index = -1
            # select adv epoch
            if args['is_poison'] and idx in args['adversary_list']:
                for temp_index in range(0, len(args['adversary_list'])):  # 0-3
                    if int(idx) == args['adversary_list'][temp_index]:
                        adversarial_index = temp_index
                        local_poison_epochs = args[str(
                            temp_index) + '_poison_epochs']
                        break

            # judge adv
            is_poison = False
            if args['is_poison'] and idx in args['adversary_list'] and (epoch in local_poison_epochs):
                is_poison = True

            # prepare dataset
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            w = local_model.update_weights(model=copy.deepcopy(global_model), features=feature_names,
                                           is_poison=is_poison, adversarial_index=adversarial_index)

            local_weights.append(w)
            local_poisoned.append(is_poison)
        # update global weights
        # , local_poisoned, alpha=1)
        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        # test auc
        pred_ans = global_model.predict(test_model_input, batch_size=256)
        logloss = log_loss(test_dataset['label'].values, pred_ans)
        aucscore = roc_auc_score(test_dataset['label'].values, pred_ans)
        print("test LogLoss", round(logloss, 4))
        print("test AUC", round(aucscore, 4))
        if aucscore > max_auc:
            # best_model = copy.deepcopy(global_model)
            min_loss = logloss
            max_auc = aucscore

        # test asr
        pred_atk = global_model.predict(poi_model_input, batch_size=256)
        atkscore = roc_auc_score(test_dataset['label'].values, pred_atk)

        atkloss = Logloss_loc(poi_dataset['label'].values, pred_atk)
        print("attack LogLoss", round(atkloss, 4))
        print("test ASR", round(atkscore, 4))

    print("|---- Min log loss: {:.4f}%".format(min_loss))
    print("|---- Best AUC: {:.4f}%".format(max_auc))
    print("test done")
