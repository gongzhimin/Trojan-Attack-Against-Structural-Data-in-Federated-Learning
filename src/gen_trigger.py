import numpy as np
import pandas as pd
import torch
import json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import SparseFeat, DenseFeat, get_feature_names, build_input_features
from deepctr_torch.inputs import combined_dnn_input

# key_to_maximize为使用select_neuron.py挑选出来的neuron序号
key_to_maximize = 34

fmap_block = dict()


def cos_sim(vec, mat):
    num = np.dot(np.array([vec]), np.array(mat).T)  # 向量点乘
    denom = np.linalg.norm(vec) * np.linalg.norm(mat, axis=1)  # 求模长的乘积
    if denom.all() == 0:
        res = 0
    else:
        res = num / denom
        res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


'''
    farward_hook用于前向传播时统计模块的输入和输出，用于统计指定层的激活值
'''


def farward_hook(module, inp, outp):
    fmap_block['in'] = inp
    fmap_block['out'] = outp


'''
    backward_hook用于在反向传播时，用ont-hot[0...1...0]替代原有梯度，使得selected neuron的上升梯度最大
    注意：这里的输入/出是对应前向传播的输入/出
    gi[0]:对bias的梯度
    gi[1]:对模块输入x的梯度
    gi[2]:对weight的梯度
    go:对模块输出的梯度
    backward_hook返回新的gi梯度进行反向传播
'''


def backward_hook(module, gi, go):
    '''
    print("------------Grad Input-----------")   #input for forwarad
    for i in range(len(gi)):
        print(gi[i].shape)   #grad for bias[43], x[1,350], weight[350,43]
    print("------------Grad Output----------")
    #print(go.shape)
    for i in range(len(go)):
       print(go[i].shape)
    print("---------------------------------")
    print("grad of fc1: ",gi[1])
'''
    targeted_grad = torch.zeros_like(gi[1])
    targeted_grad[0][key_to_maximize] = -1
    return gi[0], targeted_grad, gi[2]

# 产生trigger区域mask，为0-1矩阵


def filter_part(features):
    fields = 24 * 4 + 10
    # fields = 24 * 4 + 10 + 2
    mask = np.zeros((1, fields))
    poi_field = ['slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_score',
                 'list_time', 'device_price', 'up_life_duration']
    # trigger field
    pfield_index = []  # poison field
    pfeature_index = []  # poison feature

    for i in poi_field:
        idx = features[i][0]
        pfield_index.append(idx)
        if idx < 24:
            pfeature_index += [idx * 4 + i for i in range(4)]
        else:
            pfeature_index.append(idx - 24 + 24 * 4)

    mask[0, pfeature_index] = 1

    return mask, pfield_index


def preprocess():
    data_dir = './data/new_data.csv'
    data = pd.read_csv(data_dir)

    # sparse_features = ['uid', 'task_id', 'adv_id', 'creat_type_cd', 'adv_prim_id', 'dev_id',
    #                    'inter_type_cd', 'slot_id', 'spread_app_id', 'tags', 'app_first_class', 'app_second_class',
    #                    'city', 'city_rank', 'device_name', 'career', 'gender', 'net_type', 'residence', 'emui_dev',
    #                    'up_membership_grade', 'indu_name', 'pt_d', 'consume_purchase']

    # dense_features = ['age', 'device_size', 'his_app_size', 'his_on_shelf_time', 'app_score',
    #                   'list_time', 'device_price', 'up_life_duration', 'membership_life_duration',
    #                   'communication_avgonline_30d']
    
    sparse_features = ['uid','task_id','adv_id','creat_type_cd','adv_prim_id','dev_id',
    'inter_type_cd','slot_id','spread_app_id','tags','app_first_class','app_second_class',
    'city','city_rank','device_name','career','gender','net_type','residence','emui_dev',
    'up_membership_grade','indu_name','pt_d','consume_purchase']

    dense_features = ['age','device_size','his_app_size','his_on_shelf_time','app_score',
    'list_time','device_price','up_life_duration','membership_life_duration',
    'communication_avgonline_30d']
    # 36 features

    target = ['label']
    #  0:1 = 28.1

    # 1.Label Encoding for sparse features,and do simple Transformation for dense features
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    # 2.count #unique features for each sparse field, and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(
    ), embedding_dim=4) for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(
        linear_feature_columns + dnn_feature_columns)

    data0 = data[data["label"] == 0].iloc[0:1]

    x = {name: data0[name] for name in feature_names}

    # (type(x["uid"])) int/series

    feature_index = build_input_features(
        linear_feature_columns + dnn_feature_columns)

    if isinstance(x, dict):
        x = [x[feature] for feature in feature_index]
    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    x = torch.from_numpy(np.concatenate(x, axis=-1))

    return x, feature_index


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = torch.load("../save/smallset_cleanmodel.pth")
    model = torch.load("./save/poi.pth")
    model.eval()

    model.dnn.linears[0].register_forward_hook(farward_hook)
    model.dnn.linears[1].register_backward_hook(backward_hook)

    # get and handle input data
    x, features = preprocess()
    mask_logo, p_index = filter_part(features)

    mask = np.float32(mask_logo > 0)  # change type to float32

    #########
    sparse_embedding_list, dense_value_list = model.input_from_feature_columns(
        x, model.dnn_feature_columns, model.embedding_dict)

    x = combined_dnn_input(sparse_embedding_list, dense_value_list)
    x = torch.tensor(x, dtype=torch.float32)

    # generation trigger
    # 不同的模型stepping不同，有bn层的模型stepping要小一些
    optimizer = optim.SGD([x], lr=0)
    stepping = [(100, 1), (1500, 0.5), (1000, 0.1)][2:]
    i = 0
    obj_act = 0
    opt_act = 0
    opt_x = x.clone().detach()
    for rounds, lr in stepping:
        for r in range(rounds):
            #print("@@@@@@Round", i)
            optimizer.zero_grad()
            x = Variable(x, requires_grad=True)

            # dnn only
            output = model.out(model.dnn_linear(model.dnn(x)))
            output_fc1 = fmap_block['out']

            obj_act = output_fc1[0][key_to_maximize]
            print("selected neuron", key_to_maximize, ":", obj_act)

            if obj_act > opt_act:
                opt_act = obj_act
                opt_x = x.clone().detach()
            rank = np.argsort(-output_fc1.detach().cpu().numpy())
            print("max", rank[0][0], ":", output_fc1[0][rank[0][0]])
            if i == 0:
                before_act = output_fc1[0][key_to_maximize]
            obj = torch.tensor([[1.]]).reshape(-1)
            # obj = torch.tensor([[1.]]).reshape((1, 1))
            obj = obj.to(device)
            loss = nn.L1Loss()(output, obj.float())
            loss.backward()
            grad_mean = np.abs(x.grad.data.cpu()).mean()
            x.grad.data.mul_(torch.from_numpy(mask).to(device) * lr / (1000 * grad_mean))
            new_x = x - x.grad.data

            new_x[0, :96] = torch.clamp(new_x[0, :96], -1., 1.)
            new_x[0, 96:] = torch.clamp(new_x[0, 96:], 0., 1.)
            x = new_x

            i += 1

    print("before_act: ", before_act)
    print("after_act: ", opt_act)
    print("change of act: ", opt_act - before_act)

    # deprocess
    dict_index = {i: k for i, k in enumerate(features.keys())}
    trigger = dict()

    # Get sparse poison name
    sparse_index = [x for x in p_index if x < 24]

    p_sparse = []  # sparse name
    for i in sparse_index:
        p_sparse.append(dict_index[i])
    # print(sparse_index) [7, 8, 9, 10]
    # print(p_sparse)  ['slot_id', 'spread_app_id', 'tags', 'app_first_class']

    x = opt_x.detach().numpy()

    for i, feat in zip(sparse_index, p_sparse):
        embed = model.embedding_dict[feat]
        mat = embed.weight.clone().detach().numpy()

        vec = x[0, i*4:i*4+4]

        t = np.argmax(cos_sim(vec, mat))
        trigger[feat] = t

    # Get sparse poison name
    dense_index = [i for i in p_index if i >= 24]  # [28, 29, 30, 31]

    p_dense = []  # dense name
    for i in dense_index:
        p_dense.append(dict_index[i])

    for i, feat in zip(dense_index, p_dense):
        trigger[feat] = x[0, 96 + (i - 24)]

    print(trigger)
