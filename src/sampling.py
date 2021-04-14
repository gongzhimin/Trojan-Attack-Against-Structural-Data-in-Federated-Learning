#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import numpy as np


def sample_iid(dataset, num_users):
    """
    Sample I.I.D. client data
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)   # average
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,replace=False))
        # random but same number
        all_idxs = list(set(all_idxs) - dict_users[i]) # except selected index
    return dict_users


def sample_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data
    :param dataset:
    :param num_users:
    :return:
    """
    total_records = len(dataset)
    # 670,000 training records -->  200 records/shard * 335 shards
    num_shards, num_records = 2000, 335
    idx_shard = [i for i in range(num_shards)]   # 0-1999(2000)
    dict_users = {i: np.array([]) for i in range(num_users)} # 0- numclient


    idxs = np.arange(total_records) # 0- 670000
    labels = dataset['label'].values.flatten() # 0 1  sequence

    # sort labels # sequence 000.. 111..
    idxs_labels = np.vstack((idxs, labels))  # idx and label
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]

    idxs = idxs_labels[0, 515:] # 670515 - 670000

    # divide and assign 2 shards/client
    for i in range(num_users): # 100 users default
        rand_set = set(np.random.choice(idx_shard, 20, replace=False)) # from 2000 select 20
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_records:(rand+1)*num_records]), axis=0)
            # origin[] || new
    # print(dict_users)
    # exit(0)
    return dict_users


# def mnist_noniid_unequal(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset s.t clients
#     have unequal amount of data
#     :param dataset:
#     :param num_users:
#     :returns a dict of clients with each clients assigned certain
#     number of training records
#     """
#     # 60,000 training records --> 50 records/shard X 1200 shards
#     num_shards, num_records = 1200, 50
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_records)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # Minimum and maximum shards assigned per client:
#     min_shard = 1
#     max_shard = 30
#
#     # Divide the shards into random chunks for every client
#     # s.t the sum of these chunks = num_shards
#     random_shard_size = np.random.randint(min_shard, max_shard+1,
#                                           size=num_users)
#     random_shard_size = np.around(random_shard_size /
#                                   sum(random_shard_size) * num_shards)
#     random_shard_size = random_shard_size.astype(int)
#
#     # Assign the shards randomly to each client
#     if sum(random_shard_size) > num_shards:
#
#         for i in range(num_users):
#             # First assign each client 1 shard to ensure every client has
#             # atleast one shard of data
#             rand_set = set(np.random.choice(idx_shard, 1, replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_records:(rand+1)*num_records]),
#                     axis=0)
#
#         random_shard_size = random_shard_size-1
#
#         # Next, randomly assign the remaining shards
#         for i in range(num_users):
#             if len(idx_shard) == 0:
#                 continue
#             shard_size = random_shard_size[i]
#             if shard_size > len(idx_shard):
#                 shard_size = len(idx_shard)
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_records:(rand+1)*num_records]),
#                     axis=0)
#     else:
#
#         for i in range(num_users):
#             shard_size = random_shard_size[i]
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[i] = np.concatenate(
#                     (dict_users[i], idxs[rand*num_records:(rand+1)*num_records]),
#                     axis=0)
#
#         if len(idx_shard) > 0:
#             # Add the leftover shards to the client with minimum images:
#             shard_size = len(idx_shard)
#             # Add the remaining shard to the client with lowest data
#             k = min(dict_users, key=lambda x: len(dict_users.get(x)))
#             rand_set = set(np.random.choice(idx_shard, shard_size,
#                                             replace=False))
#             idx_shard = list(set(idx_shard) - rand_set)
#             for rand in rand_set:
#                 dict_users[k] = np.concatenate(
#                     (dict_users[k], idxs[rand*num_records:(rand+1)*num_records]),
#                     axis=0)
#
#     return dict_users



if __name__ == '__main__':
    print("test sampling")
#     dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
#                                    transform=transforms.Compose([
#                                        transforms.ToTensor(),
#                                        transforms.Normalize((0.1307,),
#                                                             (0.3081,))
#                                    ]))
#     num = 100
#     d = mnist_noniid(dataset_train, num)
