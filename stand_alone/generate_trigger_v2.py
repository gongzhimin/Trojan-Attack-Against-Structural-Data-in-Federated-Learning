import yaml
import torch
import numpy as np
import pandas as pd

import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from deepctr_torch.models import DeepFM
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from deepctr_torch.inputs import get_feature_names, build_input_features, SparseFeat, DenseFeat, combined_dnn_input


with open(r"./stand_alone/stand_alone_params.yaml", 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)["poison_rate_com"]

key_to_maximize = CONFIG["selected_neuron"]
poi_field = CONFIG["mask_fields"]
data_dir = CONFIG["data_dir"]
sparse_features = CONFIG["sparse_features"]
dense_features = CONFIG["dense_features"]
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
fmap_block = dict()


def cos_sim(vec, mat):
    num = np.dot(np.array(vec), np.array(mat).T)
    denom = np.linalg.norm(vec) * np.linalg.norm(mat, axis=1)
    if denom.all() == 0:
        res = 0
    else:
        res = num / denom

    return 0.5 + 0.5 * res


def farward_hook(module, inp, outp):
    fmap_block["in"] = inp
    fmap_block["out"] = outp


def backward_hook(module, gi, go):
    targeted_grad = torch.zeros_like(gi[1])
    targeted_grad[0][key_to_maximize] = -1

    return gi[0], targeted_grad, gi[2]


def filter_part(features):
    fields = 24 * 4 + 10
    mask = np.zeros((1, fields))
    pfield_index = []
    pfeature_index = []

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
    data = pd.read_csv(data_dir, sep='|')
    del data["communication_onlinerate"]
    data = pd.DataFrame(data, dtype="int32")

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])

    fixlen_feature_columns = [SparseFeat(feat,
                                         vocabulary_size=data[feat].nunique(),
                                         embedding_dim=4)
                              for i, feat in enumerate(sparse_features)] + [DenseFeat(feat, 1, ) for feat in dense_features]

    dnn_feature_columns = fixlen_feature_columns
    linear_feature_columns = fixlen_feature_columns

    feature_names = get_feature_names(linear_feature_columns +
                                      dnn_feature_columns)

    data0 = data[data["label"] == 0].iloc[0:1]

    x = {name: data0[name] for name in feature_names}

    feature_index = build_input_features(linear_feature_columns +
                                         dnn_feature_columns)

    if isinstance(x, dict):
        x = [x[feature] for feature in feature_index]

    for i in range(len(x)):
        if len(x[i].shape) == 1:
            x[i] = np.expand_dims(x[i], axis=1)

    x = torch.from_numpy(np.concatenate(x, axis=-1))

    model = DeepFM(linear_feature_columns, dnn_feature_columns,
                   task="binary", device=device)
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", "binary_crossentropy"])
    state_dict = torch.load(
        "./save/clear_model_{}.pth".format(CONFIG["epochs"]))
    model.load_state_dict(state_dict)

    x = torch.where(torch.isnan(x), torch.full_like(x, 0), x)

    return x, feature_index, model


if __name__ == "__main__":
    x, features, model = preprocess()
    model.eval()

    model.dnn.linears[0].register_forward_hook(farward_hook)
    model.dnn.linears[1].register_backward_hook(backward_hook)

    mask_logo, p_index = filter_part(features)

    mask = np.float32(mask_logo > 0)

    sparse_embedding_list, dense_value_list = model.input_from_feature_columns(
        x.float(), model.dnn_feature_columns, model.embedding_dict)

    x = combined_dnn_input(sparse_embedding_list, dense_value_list)
    x = torch.tensor(x, dtype=torch.float32)

    optimizer = optim.SGD([x], lr=0)
    stepping = [(100, 1), (1500, 0.5), (1000, 0.1)][2:]
    i = 0
    obj_act = 0
    opt_act = 0
    opt_x = x.clone().detach()

    for rounds, lr in stepping:
        for r in range(rounds):
            optimizer.zero_grad()
            x = Variable(x, requires_grad=True)

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

            obj = torch.tensor([[1.]]).reshape((1, 1))
            obj = obj.to(device)
            loss = nn.L1Loss()(output, obj.float())
            loss.backward()
            grad_mean = np.abs(x.grad.data.cpu()).mean()
            x.grad.data.mul_(torch.from_numpy(mask).to(device) * lr / (1000 * grad_mean + 1e-16))
            new_x = x - x.grad.data

            new_x[0, :96] = torch.clamp(new_x[0, :96], -1., 1.)
            new_x[0, 96:] = torch.clamp(new_x[0, 96:], 0., 1.)
            x = new_x

            i += 1
    
    print("before_act: ", before_act)
    print("after_act: ", opt_act)
    print("change of act: ", opt_act - before_act)

    dict_index = {i: k for i, k in enumerate(features.keys())}
    trigger = dict()

    sparse_index = [x for x in p_index if x < 24]

    p_sparse = []  # sparse name
    for i in sparse_index:
        p_sparse.append(dict_index[i])

    x = opt_x.detach().numpy()

    for i, feat in zip(sparse_index, p_sparse):
        embed = model.embedding_dict[feat]
        mat = embed.weight.clone().detach().numpy()

        vec = x[0, i*4:i*4+4]

        t = np.argmax(cos_sim(vec, mat))
        trigger[feat] = t

    dense_index = [i for i in p_index if i >= 24]  # [28, 29, 30, 31]

    p_dense = []  # dense name
    for i in dense_index:
        p_dense.append(dict_index[i])

    for i, feat in zip(dense_index, p_dense):
        trigger[feat] = x[0, 96 + (i - 24)]

    print(trigger)
