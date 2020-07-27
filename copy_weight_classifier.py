import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
# from sklearn.feature_selection import SelectKBest, chi2

# copy_weight_classifier
def multi_to_single_label(label_start, label_end):

    # added at the end for single labels
    label_column = None
    # # added at the end for weights
    label_weights = None

    for index, row in DATASET.iterrows():
        # label1_value, label2_value, label3_value = row['Label 1'], row['Label 2'], row['Label 3']
        label_value_list = []
        for label in range(label_start, label_end+1):
            label_value_list.append(row[label])
        # reweighting condition for entropy based label assignment
        if(sum(label_value_list) > 1):
            sum_of_list = sum(label_value_list)
            for label in range(0,len(label_value_list)):
                # label_value_list[label] = label_value_list[label]/sum_of_list
                column_name = DATASET.columns[label+label_start]
                DATASET.loc[index, column_name] = label_value_list[label]/sum_of_list

    SINGLE_DATASET = DATASET.copy()

    DATASET.insert(label_end+1,'Weight',label_column)
    DATASET.insert(label_end+1,'Label',label_weights)

    SINGLE_DATASET.insert(label_end+1,'Weight',label_column)
    SINGLE_DATASET.insert(label_end+1,'Label',label_weights)

    # print(DATASET)
    # print(SINGLE_DATASET)

    SINGLE_DATASET = SINGLE_DATASET[0:0]

    # create new single label dataset with multi class (multilabel to singlelabel)
    for index, row in DATASET.iterrows():
        label_name = ''
        weight_to_copy = 0
        for label in range(label_start, label_end+1):
            if(row[label] > 0):
                newrow = row
                label_name = DATASET.columns[label]
                weight_to_copy = row[label]
                # row.to_dict()
                # print(row)
                newrow['Label'] = label_name
                newrow['Weight'] = weight_to_copy
                SINGLE_DATASET = SINGLE_DATASET.append(newrow)
                # SINGLE_DATASET.loc[index, 'Label'] = label_name
                # SINGLE_DATASET.loc[index, 'Weight'] = weight_to_copy

    # print(SINGLE_DATASET)

    # label_start = SINGLE_DATASET.columns.get_loc('Label 1')
    # label_end = SINGLE_DATASET.columns.get_loc('Label 3')
    for label in range(label_start, label_end+1):
        SINGLE_DATASET = SINGLE_DATASET.drop(SINGLE_DATASET.columns[label_start], axis=1)
    print(SINGLE_DATASET)

if __name__ == '__main__':
    DATASET = pd.read_csv("C:\\Users\\MOHNISH\\AI\\Research_work\\ELA-CHI\\dataset.csv")

    print(DATASET)
    """
    label start index and label end index is defined in order
    to convert multilabel to multiclass (i.e single label dataset)
    """
    label_start = DATASET.columns.get_loc('Label 1')
    label_end = DATASET.columns.get_loc('Label 3')

    multi_to_single_label(label_start, label_end)
