import os
import time
import ast

import pandas as pd
import pickle

from bta_cnn_iterator.cnn_initializer import run_cnn_initializer
from bta_cnn_iterator.bta_v2 import update_blocking_cand, run_bta_blocking, run_bta_matching
from bta_cnn_iterator.cnn import run_cnn

# global variable
# put the datasets under './experiment_data/'
TAB_NAME = 'NYC'
COL_NAME = "Catalog Names"
BUS_TERM = "Bus Terms"
DESCR = "Bus Terms"


if __name__ == "__main__":
    start_time = time.time()

    # load whole table (gold)
    catalog = pd.read_csv('experiment_data/gold.csv')
    col_table, bt_table, gold, inverted_index = run_cnn_initializer(catalog, TAB_NAME, COL_NAME, BUS_TERM, DESCR)

    train_labels = None
    epoch = 0
    n = 10
    # while epoch < 70: # for revision
    while epoch < 1:
        epoch_start_time = time.time()

        print("Current epoch: " + str(epoch))
        # num_iteration = 1 # for revision
        num_iteration = 70 #150
        if epoch < 2:
            if epoch == 0:
                save = True
                features = None
            else:
                save = False
                features = pd.concat([blocking_cand[[COL_NAME, BUS_TERM]], features], axis=1) #concate side to side horizontally

            blocking_cand = run_bta_blocking(gold, train_labels, TAB_NAME, COL_NAME, BUS_TERM, n, col_table, bt_table, inverted_index)

            if epoch > 0:  # saving featurization time by reusing previously calculated features
                blocking_cand = blocking_cand.merge(features, on=[COL_NAME, BUS_TERM], how="left")

            matching_output, features, train_labels = run_bta_matching(blocking_cand, gold,
                                                                       features, train_labels, num_iteration, epoch,
                                                                       TAB_NAME, COL_NAME, BUS_TERM, n)

        else:
            blocking_cand = update_blocking_cand(blocking_cand, col_table)
            matching_output, _, train_labels = run_bta_matching(blocking_cand, gold, features, train_labels,
                                                                num_iteration, epoch, TAB_NAME, COL_NAME, BUS_TERM, n)

        # col_table = run_cnn(matching_output, col_table, gold, TAB_NAME, COL_NAME, BUS_TERM) # for revision

        if (len(blocking_cand) - len(train_labels)) < 10:
            break

        epoch = epoch + 1

        epoch_end_time = time.time()
        print(epoch_end_time - epoch_start_time)

    end_time = time.time()
    print(end_time - start_time)
    print("Total number of training examples: " + str(len(train_labels)))
    train_labels = train_labels.merge(col_table, on=["col_id"])
    train_labels = train_labels[[COL_NAME, "bt_id", "Label"]]
    train_labels = train_labels.merge(bt_table, on=["bt_id"])
    train_labels = train_labels[[COL_NAME, BUS_TERM, "Label"]]
    train_labels = train_labels.merge(gold, on=[COL_NAME, BUS_TERM])
    # train_labels = train_labels.merge(gold, on=["col_id", "bt_id"])
    print("Total number of matches (gold): " + str(len(gold)))
    print("Matches in training examples: " + str(len(train_labels)))
    print("done")
