import copy
import math
import time
import yaml
import torch
import random
import numpy as np
import pandas as pd

from deepctr_torch.models import DeepFM
from sklearn.metrics import roc_auc_score
from deepctr_torch.inputs import get_feature_names


with open(r"./stand_alone/stand_alone_params.yaml", 'r') as f:
    PARAMS = yaml.load(f, Loader=yaml.FullLoader)["poison_rate_con"]


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


def poison_data(dataset, mask, poison_rate=0.2):
    poisoned_dataset = copy.deepcopy(dataset)
    poison_batch = int(len(poisoned_dataset) * poison_rate)
    idx = poisoned_dataset.axes[0].values.tolist()
    poisoned_idx = random.sample(idx, poison_batch)

    for field, value in mask.items():
        poisoned_dataset.loc[poisoned_idx, field] = value

    return poisoned_dataset


def generate_dataset(data_dir, sparse_features, dense_features):
    import pandas as pd
    data = pd.read_csv(data_dir)

    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    label_encoder = LabelEncoder()
    min_max_scaler = MinMaxScaler(feature_range=(0, 1))
    for feature in sparse_features:
        data[feature] = label_encoder.fit_transform(data[feature])
    data[dense_features] = min_max_scaler.fit_transform(data[dense_features])

    from sklearn.model_selection import train_test_split
    train_set, test_set = train_test_split(data, test_size=0.2,
                                           random_state=2018)

    from deepctr_torch.inputs import SparseFeat, DenseFeat
    spare_feature_columns = [SparseFeat(feature,
                                        vocabulary_size=data[feature].nunique(), embedding_dim=4)
                             for i, feature in enumerate(sparse_features)]
    dense_feature_columns = [DenseFeat(feature, 1)
                             for feature in dense_features]
    feature_columns = spare_feature_columns + dense_feature_columns

    return (train_set, test_set), feature_columns


if __name__ == "__main__":
    data_dir = PARAMS["data_dir"]
    sparse_features = PARAMS["sparse_features"]
    dense_features = PARAMS["dense_features"]
    (train_set, test_set), feature_columns = generate_dataset(data_dir,
                                                              sparse_features, dense_features)
    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns
    feature_names = get_feature_names(feature_columns)

    assert PARAMS["model"].lower() == "deepfm", "No DeepFM"
    group_name = PARAMS["group_name"]
    poison_rate = PARAMS["poison_rate"][1]
    if group_name == "clear":   # pre-train the model
        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task="binary")
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["binary_crossentropy"])
        train_model_input = {name: train_set[name] for name in feature_names}

    elif group_name == "model_dependent" or group_name == "random":
        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task="binary")
        state_dict = torch.load("./save/clear_model.pth")
        model.load_state_dict(state_dict)
        if group_name == "model_dependent":
            trigger = PARAMS["trigger"]
        else:
            trigger = PARAMS["random_mask"]
        poisoned_train_set = poison_data(train_set, trigger, poison_rate)
        train_model_input = {name: poisoned_train_set[name]
                             for name in feature_names}
    else:
        raise Exception("No such group: {}".format(group_name))

    model.train()
    test_model_input = {name: test_set[name]
                        for name in feature_names}

    epochs = PARAMS["epochs"]
    batch_size = PARAMS["batch_size"]
    validation_split = PARAMS["validation_split"]
    history = model.fit(x=train_model_input,
                        y=train_set["label"].values,
                        batch_size=batch_size, epochs=epochs,
                        validation_split=validation_split)
    if group_name == "clear":
        torch.save(model.state_dict(), "./save/clear_model.pth")

    # test_pred = model.predict(test_model_input)
    # test_logloss = logloss_loc(test_set["label"].values, test_pred)
    # test_auc_score = roc_auc_score(test_set["label"].values, test_pred)

    # poisoned_test_set = poison_data(test_set, trigger, poison_rate)
    # poisoned_model_input = {name: poisoned_test_set[name]
    #                         for name in feature_names}
    # attack_pred = model.predict(poisoned_model_input)
    # attack_logloss = logloss_loc(poisoned_test_set["label"].values,
    #                              attack_pred)
    # attack_auc_score = roc_auc_score(poisoned_test_set["label"].values,
    #                                  attack_pred)
