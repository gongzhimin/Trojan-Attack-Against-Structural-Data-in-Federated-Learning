#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import random
import pandas as pd
from sklearn.utils import shuffle


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader = self.train_val_test(dataset, list(idxs))
        self.device = 'cuda' if args['gpu'] else 'cpu'


    def train_val_test(self, dataset, idxs):
        """
        Returns train dataloaders for a given dataset
        and user indexes.
        """

        index = [int(i) for i in idxs]
        trainloader = dataset.iloc[index,:]
        return copy.deepcopy(trainloader)

    def update_weights(self, model, features, is_poison=False, adversarial_index=-1):
        # Set optimizer for the local updates
        model.compile("adam", "binary_crossentropy", metrics=['binary_crossentropy'], )

        if is_poison:
            poi = copy.deepcopy(self.trainloader)

            poison_batch = int(len(self.trainloader) * self.args['poison_rate'])
            namelist = self.trainloader._stat_axis.values.tolist()
            poison_idx = random.sample(namelist, poison_batch)

            poison_patterns = self.args[str(adversarial_index) + '_poison_pattern']
            poi.loc[poison_idx, 'label'] = int(1)  # ctr target

            trigger = {'slot_id': 6, 'spread_app_id': 66, 'tags': 27,
                       'app_first_class': 0, 'app_score': 1.0, 'list_time': 1.0,
                       'device_price': 1.0, 'up_life_duration': 1.0}


            for feat in poison_patterns:
                # poi.loc[poison_idx, feat] = 0#trigger[feat]
                poi.loc[poison_idx, feat] = trigger[feat]

            poi = poi.loc[poison_idx, :]
            self.trainloader = pd.concat([self.trainloader, poi]) # concatenate clean&poi data
            self.trainloader = shuffle(self.trainloader) # shuffle


        train_model_input = {name: self.trainloader[name] for name in features}

        history = model.fit(train_model_input, self.trainloader['label'].values,
                            batch_size=self.args['local_bs'], epochs=self.args['local_ep'],
                            verbose=self.args['verbose'], validation_split=0.1,)

        return model.state_dict()



    # def add_trigger(self, index, adversarial_index):
    #     # poison_patterns = []
    #     # if adversarial_index == -1:
    #     #     for i in range(0, self.params['trigger_num']):
    #     #         poison_patterns = poison_patterns + self.params[str(i) + '_poison_pattern']
    #     # else:
    #     poison_patterns = self.args[str(adversarial_index) + '_poison_pattern']
    #
    #     # for i in range(0, len(poison_patterns)):
    #     #     pos = poison_patterns[i]
    #     self.trainloader.loc[index, poison_patterns] = 0
