import pandas as pd
import torch
import copy
import random
import math
import numpy as np

from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names



def Logloss_loc(y_true, y_pred):
    eps = 1e-15

    # Prepare numpy array data
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    assert (len(y_true) and len(y_true) == len(y_pred))

    # Clip y_pred between eps and 1-eps
    p = np.clip(y_pred, eps, 1-eps)
    loss = 0.0
    for i in range(len(y_true)):
        loss += (- y_true[i] * math.log(p[i]) - (1 - y_true[i]) * math.log(1-p[i]))

    return loss / len(y_true)


def poison_data(test_dataset, prate=0.1):

    poi_dataset = copy.deepcopy(test_dataset)
    poison_patterns = ['age', 'device_size',
                      'his_app_size', 'his_on_shelf_time', 'app_score', 'list_time']
    poison_batch = int(len(poi_dataset) * prate)
    namelist = poi_dataset._stat_axis.values.tolist()
    poison_idx = random.sample(namelist, poison_batch)
    poi_dataset.loc[poison_idx, target] = int(1)

    # trigger ={'slot_id': 9, 'spread_app_id': 31, 'tags': 23,
    #           'app_first_class': 1, 'app_score': 1.0, 'list_time': 1.0,
    #           'device_price': 1.0, 'up_life_duration': 0.0}


    for feat in poison_patterns:
        poi_dataset.loc[poison_idx, feat] = 0 #0 for random or trigger[feat]
    return poi_dataset


if __name__ == "__main__":
    is_poison = True

    data_dir = '../data/new_data.csv'
    data = pd.read_csv(data_dir)

    sparse_features = ['uid','task_id','adv_id','creat_type_cd','adv_prim_id','dev_id',
    'inter_type_cd','slot_id','spread_app_id','tags','app_first_class','app_second_class',
    'city','city_rank','device_name','career','gender','net_type','residence','emui_dev',
    'up_membership_grade','indu_name','pt_d','consume_purchase']

    dense_features = ['age','device_size','his_app_size','his_on_shelf_time','app_score',
    'list_time','device_price','up_life_duration','membership_life_duration',
    'communication_avgonline_30d']


    target = ['label']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # o2 count #unique features for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(),embedding_dim=4 )
                            for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                            for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    # o3 generate input data for model

    train, test = train_test_split(data, test_size=0.2, random_state=2018)

    # test data for auc
    test_model_input = {name: test[name] for name in feature_names}

    # poi test data for attack
    if is_poison:
        print("poison test data")
        # logloss
        poi_dataset = poison_data(test_dataset)
        poi_model_input_ls = {name: poi_dataset[name] for name in feature_names}
        plabel_ls = copy.deepcopy(poi_dataset[target].values)
        
        # trigger + 所有clean样本（所谓的正例），所有的真正的负例
        neg = test_dataset.loc[test_dataset[target] == 0]
        poi_dataset = poison_data(test_dataset)
        poi = pd.concat([neg, poi_dataset])
        poi = shuffle(poi)
        poi_model_input_a1 = {name: poi[name] for name in feature_names}
        plabel_a1 = copy.deepcopy(poi[target].values)

        # trigger + 所有clean负例（所谓的正例），所有的真正的负例
        poi_dataset = poison_data(neg)
        poi = pd.concat([neg, poi_dataset])
        poi = shuffle(poi)
        poi_model_input_a2 = {name: poi[name] for name in feature_names}
        plabel_a2 = copy.deepcopy(poi[target].values)

        # poison training data
        poi_dataset = poison_data(train)
        train = pd.concat([train, poi])
        train = shuffle(train)

    train_model_input = {name: train[name] for name in feature_names} 
    test_model_input = {name: test[name] for name in feature_names}

    # o4 Define Model,train,predict and evaluate
    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
    model.compile("adam", "binary_crossentropy",
                    metrics=['binary_crossentropy'], )
    # load pretrained model
    # model.load_state_dict(torch.load("clean.pth"), strict=False)
    # model = torch.load("cleanmodel.pth")

    history = model.fit(train_model_input, train[target].values,
                        batch_size=1024, epochs=10, verbose=2, validation_split=0.1, )


    pred_ans = model.predict(test_model_input, batch_size=256)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

    if is_poison:
        # test asr
        pred_atk = global_model.predict(poi_model_input_ls, batch_size=256)
        atkloss = Logloss_loc(plabel_ls, pred_atk)
        print("attack LogLoss", round(atkloss, 4))

        ########## asr1

        pred_atk = global_model.predict(poi_model_input_a1, batch_size=256)
        atkscore = roc_auc_score(plabel_a1, pred_atk)
        print("test ASR1", round(atkscore, 4))

        ######### asr2
        pred_atk = global_model.predict(poi_model_input_a2, batch_size=256)
        atkscore = roc_auc_score(plabel_a2, pred_atk)
        print("test ASR2", round(atkscore, 4))
