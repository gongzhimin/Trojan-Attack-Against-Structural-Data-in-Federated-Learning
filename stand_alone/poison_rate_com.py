import os
import copy
import math
import time
import json
import yaml
import torch
import random
import datetime
import numpy as np
import pandas as pd

from sklearn.utils import shuffle
from deepctr_torch.models import DeepFM
from sklearn.metrics import roc_auc_score
from deepctr_torch.inputs import get_feature_names

from utils import logloss_loc, poison_data, generate_dataset, capture_cmdline, choose_device

# To run background:
# nohup python stand_alone/poison_rate_com.py > poison_rate_com.log 2>&1 &

with open(r"./stand_alone/stand_alone_params.yaml", 'r') as f:
    PARAMS = yaml.load(f, Loader=yaml.FullLoader)["poison_rate_com"]


if __name__ == "__main__":
    PARAMS = capture_cmdline(PARAMS)
    data_dir = PARAMS["data_dir"]
    sparse_features = PARAMS["sparse_features"]
    dense_features = PARAMS["dense_features"]
    (train_set, test_set), feature_columns = generate_dataset(data_dir,
                                                              sparse_features, dense_features)
    del data_dir, sparse_features, dense_features

    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns
    feature_names = get_feature_names(feature_columns)
    del feature_columns

    assert PARAMS["model"].lower() == "deepfm", "No DeepFM"
    group_name = PARAMS["group_name"]
    poison_rate = PARAMS["poison_rate"][1]
    device = choose_device(PARAMS)
    if group_name == "clear":   # pre-train the model
        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task="binary", device=device)
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy", "binary_crossentropy"])
        train_model_input = {name: train_set[name] for name in feature_names}

    elif group_name == "model_dependent" or group_name == "random":
        model = DeepFM(linear_feature_columns, dnn_feature_columns,
                       task="binary", device=device)
        model.compile(optimizer="adam", loss="binary_crossentropy",
                      metrics=["accuracy", "binary_crossentropy"])
        state_dict = torch.load(
            "./save/clear_model_{}.pth".format(PARAMS["epochs"]))
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

    del linear_feature_columns, dnn_feature_columns

    epochs = PARAMS["epochs"]
    batch_size = PARAMS["batch_size"]
    validation_split = PARAMS["validation_split"]

    # record the metrics with json dictionary
    results_dir = PARAMS["results_dir"]
    if not os.path.exists(results_dir):
        results_dict = {
            "model": PARAMS["model"],
            "dataset": PARAMS["dataset"],
            "mask_size": PARAMS["mask_size"],
            "mask_fields": PARAMS["mask_fields"],
            "results": []
        }
        result = {
            "selected_neuron": PARAMS["selected_neuron"],
            "poison_rate": poison_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "train_size": len(train_set),
            "test_size": len(test_set),
            "clear": {},
            "random": {},
            "model_dependent": {}
        }
        results_dict["results"].append(result)

    else:
        with open(results_dir, 'r') as f:
            results_dict = json.load(f)
        result = [result for result in results_dict["results"]
                  if result["poison_rate"] == poison_rate and
                  result["batch_size"] == batch_size and
                  result["epochs"] == epochs and
                  result["train_size"] == len(train_set) and
                  result["test_size"] == len(test_set)]
        if len(result) == 0:
            result = {
                "selected_neuron": PARAMS["selected_neuron"],
                "poison_rate": poison_rate,
                "batch_size": batch_size,
                "epochs": epochs,
                "train_size": len(train_set),
                "test_size": len(test_set),
                "clear": {},
                "random": {},
                "model_dependent": {}
            }
            results_dict["results"].append(result)
        else:
            result = result[0]

    if group_name == "clear":
        label_values = train_set["label"].values
    else:
        label_values = poisoned_train_set["label"].values
    model.train()
    history_op = model.fit(x=train_model_input,
                           y=label_values,
                           batch_size=batch_size, epochs=epochs,
                           validation_split=validation_split)
    del train_model_input, label_values, train_set
    if group_name == "clear":
        torch.save(model.state_dict(),
                   "./save/clear_model_{}.pth".format(PARAMS["epochs"]))

    test_model_input = {name: test_set[name] for name in feature_names}
    test_pred = model.predict(test_model_input)
    del test_model_input
    test_logloss = logloss_loc(test_set["label"].values, test_pred)
    test_auc_score = roc_auc_score(test_set["label"].values, test_pred)
    del test_pred

    history = history_op.history
    result[group_name]["train_loss"] = history["binary_crossentropy"]
    result[group_name]["train_accuracy"] = history["accuracy"]
    result[group_name]["val_loss"] = history["val_binary_crossentropy"]
    result[group_name]["val_accuracy"] = history["val_accuracy"]
    del history_op
    result[group_name]["test_logloss"] = test_logloss
    result[group_name]["test_auc_score"] = test_auc_score

    if group_name != "clear":
        poisoned_test_set = poison_data(test_set, trigger, poison_rate)

        poisoned_logloss_input = {name: poisoned_test_set[name]
                                  for name in feature_names}
        label_logloss = copy.deepcopy(poisoned_test_set["label"].values)
        pred_logloss = model.predict(poisoned_logloss_input)
        del poisoned_logloss_input
        attack_logloss = logloss_loc(label_logloss, pred_logloss)
        del label_logloss, pred_logloss

        negative_set = test_set.loc[test_set["label"] == 0]
        del test_set
        poisoned_asr1_set = pd.concat([negative_set, poisoned_test_set])
        del poisoned_test_set
        poisoned_asr1_set = shuffle(poisoned_asr1_set)
        poisoned_asr1_input = {name: poisoned_asr1_set[name]
                               for name in feature_names}
        label_asr1 = copy.deepcopy(poisoned_asr1_set["label"].values)
        del poisoned_asr1_set
        pred_asr1 = model.predict(poisoned_asr1_input)
        asr1 = roc_auc_score(label_asr1, pred_asr1)
        del label_asr1, pred_asr1

        poisoned_negative_set = poison_data(negative_set, trigger, poison_rate)
        poisoned_asr2_set = pd.concat([negative_set, poisoned_negative_set])
        del negative_set, poisoned_negative_set
        poisoned_asr2_set = shuffle(poisoned_asr2_set)
        poisoned_asr2_input = {name: poisoned_asr2_set[name]
                               for name in feature_names}
        label_asr2 = copy.deepcopy(poisoned_asr2_set["label"].values)
        del poisoned_asr2_set
        pred_asr2 = model.predict(poisoned_asr2_input)
        asr2 = roc_auc_score(label_asr2, pred_asr2)
        del label_asr2, pred_asr2

        result[group_name]["trigger"] = trigger
        result[group_name]["attack_logloss"] = attack_logloss
        result[group_name]["asr1"] = asr1
        result[group_name]["asr2"] = asr2

    with open(results_dir, 'w') as f:
        json.dump(results_dict, f, ensure_ascii=False)
