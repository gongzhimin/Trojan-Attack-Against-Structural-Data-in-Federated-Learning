import sys
import copy
import math
import torch
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from deepctr_torch.inputs import SparseFeat, DenseFeat
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def logloss_loc(y_true, y_pred):
    eps = 1e-15
    loss_sum = 0.0
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert len(y_true) and len(y_true) == len(y_pred), "||y_true||!=||y_pred||"
    y_pred = np.clip(y_pred, eps, 1 - eps)

    for i in range(len(y_true)):
        loss_sum += - y_true[i] * math.log(y_pred[i]) - (
            1 - y_true[i]) * math.log(1 - y_pred[i])
    loss = loss_sum / len(y_pred)

    return loss


def poison_data(dataset, trigger, poison_rate):
    poisoned_dataset = copy.deepcopy(dataset)
    poison_batch = int(len(poisoned_dataset) * poison_rate)
    idx = poisoned_dataset.axes[0].values.tolist()
    poisoned_idx = random.sample(idx, poison_batch)
    poisoned_dataset.loc[poisoned_idx, "label"] = int(1)

    for field, value in trigger.items():
        poisoned_dataset.loc[poisoned_idx, field] = value

    return poisoned_dataset


def generate_dataset(data_dir, sparse_features, dense_features):
    data = pd.read_csv(data_dir, sep='|')
    # data = pd.read_csv(data_dir)
    # data = pd.read_csv(data_dir, dtype="int32")

    for feature in sparse_features:
        label_encoder = LabelEncoder()
        data[feature] = label_encoder.fit_transform(data[feature])
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = min_max_scaler.fit_transform(data[dense_features])

    train_set, test_set = train_test_split(data, test_size=0.2,
                                           random_state=2018)

    spare_feature_columns = [SparseFeat(feature,
                                        vocabulary_size=data[feature].nunique(), embedding_dim=4)
                             for i, feature in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feature, 1, )
                             for feature in dense_features]
    feature_columns = spare_feature_columns + dense_feature_columns

    return (train_set, test_set), feature_columns


def capture_cmdline(params):
    if len(sys.argv) == 1:
        return params
        
    group_name = sys.argv[1]
    params["group_name"] = group_name

    poison_rate = float(sys.argv[2])
    params["poison_rate"] = poison_rate

    trigger_size = float(sys.argv[3])
    params["trigger_size"] = trigger_size

    params["mask_fields"] = map_mask_fields(trigger_size)
    params["random_trigger"] = generate_random_trigger(params["mask_fields"])

    print("group: ", params["group_name"], " | "
          "poison rate: ", params["poison_rate"], " | "
          "trigger size: ", params["trigger_size"])

    return params


def choose_device(params):
    device = params["device"]
    use_cuda = params["use_cuda"]

    if use_cuda and torch.cuda.is_available():
        print("cuda ready...")
        device = "cuda:0"

    return device


def map_mask_fields(trigger_size):
    if trigger_size == 0.2:
        mask_fields = ['consume_purchase', 'app_first_class', 'up_life_duration',
                       'pt_d', 'city', 'his_on_shelf_time', 'communication_avgonline_30d']

    elif trigger_size == 0.15:
        mask_fields = ['tags', 'dev_id',
                       'up_life_duration', 'his_app_size', 'list_time']

    elif trigger_size == 0.1:
        mask_fields = ['device_price',
                       'membership_life_duration', 'device_name']

    elif trigger_size == 0.03:
        mask_fields = ['device_price']
        # raise Exception("Mask fields with size 0.03 haven't been specified!")

    else:
        raise Exception("No such trigger size!")

    return mask_fields


def generate_random_trigger(mask_fields):
    random_trigger = {}
    for field in mask_fields:
        random_trigger[field] = 0

    return random_trigger


def generate_model_dependent_trigger():
    pass
