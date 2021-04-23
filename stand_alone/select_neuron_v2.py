import yaml
import torch
import numpy as np
from deepctr_torch.models import DeepFM
from deepctr_torch.inputs import get_feature_names

from utils import generate_dataset


with open(r"./stand_alone/stand_alone_params.yaml", 'r') as f:
    CONFIG = yaml.load(f, Loader=yaml.FullLoader)["poison_rate_com"]


if __name__ == "__main__":
    data_dir = CONFIG["data_dir"]
    sparse_features = CONFIG["sparse_features"]
    dense_features = CONFIG["dense_features"]
    (train_set, test_set), feature_columns = generate_dataset(data_dir,
                                                              sparse_features, dense_features)
    del data_dir, sparse_features, dense_features

    dnn_feature_columns = feature_columns
    linear_feature_columns = feature_columns
    feature_names = get_feature_names(feature_columns)
    del feature_columns

    model = DeepFM(linear_feature_columns, dnn_feature_columns,
                   task="binary")
    model.compile(optimizer="adam", loss="binary_crossentropy",
                  metrics=["accuracy", "binary_crossentropy"])
    state_dict = torch.load(
        "./save/clear_model_{}.pth".format(CONFIG["epochs"]))
    model.load_state_dict(state_dict)

    params = {}
    for name, parameters in model.named_parameters():
        print(name, ":", parameters.size())
        params[name] = parameters.cpu().detach().numpy()

    # sort the neurons in fc0 layer by sum of |weight|
    rank = np.argsort(params["dnn.linears.0.weight"].sum(axis=1))
    print("rank the neurons in fc0 layer by sum of |weight|:")
    print(rank)
