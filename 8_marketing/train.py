# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import pandas as pd

# Prepares the input data.
input_data = pd.read_csv('./train.csv', index_col='row_id')
