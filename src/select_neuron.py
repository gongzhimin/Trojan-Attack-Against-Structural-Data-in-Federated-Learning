import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data.dataloader as dataloader
import torch.nn as nn
import numpy as np
from deepctr_torch.models import DeepFM

if __name__ == "__main__":

    # model = torch.load("./save/smallset_cleanmodel.pth")
    model = torch.load("./save/poi.pth")
    model.eval()


    parm={}
    for name, parameters in model.named_parameters():
        print(name,":",parameters.size())
        parm[name]=parameters.cpu().detach().numpy()

    print("test mid")
    #print("top1 weight sum of fc1:",np.argsort(-parm['fc1.weight'].sum(axis=1))[0])
    # print(parm['dnn.linears.0.weight'].shape) [256,106]
    weight = parm['dnn.linears.0.weight']
    print(np.argsort(-abs(parm['dnn.linears.0.weight'].sum(axis=1))))
    print("top1 weight sum of fc1:",np.argsort(-parm['dnn.linears.0.weight'].sum(axis=1))[0])
    # neuron = 189
