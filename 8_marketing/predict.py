# -*- coding: utf-8 -*-

import pandas as pd

# Prepares the test data.
input_test = pd.read_csv('1_test.csv', index_col='row_id')
test_data  = pd.merge(input_test, input_users, on='user_id', how='inner')

# Replaces some hard-to-handle values.
test_data['last_open_day']     = pd.to_numeric(test_data['last_open_day'].replace(['Never open'], '20000'))
test_data['last_login_day']    = pd.to_numeric(test_data['last_login_day'].replace(['Never login'], '1000'))
test_data['last_checkout_day'] = pd.to_numeric(test_data['last_checkout_day'].replace(['Never checkout'], '2000'))
test_data['age']               = test_data['age'].abs()

# Drops some columns.
test_data = test_data.drop(columns=['grass_date', 'user_id', 'attr_1', 'attr_2'])

# Predicts the data.
result = pipeline.predict(test_data.head())

# Predicts the data.
result = pipeline.predict(test_data.head())

# Saves the result.
pd.DataFrame({'open_flag': result}).to_csv('result.csv', index_label='row_id')
