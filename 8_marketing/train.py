# -*- coding: utf-8 -*-

from sklearn.compose         import ColumnTransformer
from sklearn.ensemble        import RandomForestRegressor
from sklearn.pipeline        import Pipeline
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import Normalizer
from sklearn.preprocessing   import OneHotEncoder
import numpy  as np
import pandas as pd

# Prepares the input data.
input_train  = pd.read_csv('0_train.csv', index_col='row_id')
input_users = pd.read_csv('2_users.csv')
input_data = pd.merge(input_train, input_users, on='user_id', how='inner')

# Replaces some hard-to-handle values.
input_data['last_open_day']     = pd.to_numeric(input_data['last_open_day'].replace(['Never open'], '20000'))
input_data['last_login_day']    = pd.to_numeric(input_data['last_login_day'].replace(['Never login'], '1000'))
input_data['last_checkout_day'] = pd.to_numeric(input_data['last_checkout_day'].replace(['Never checkout'], '2000'))
input_data['age']               = input_data['age'].abs()

# Splits the training set & validation set.
input_X = input_data.drop(columns=['open_flag', 'grass_date', 'user_id', 'attr_1', 'attr_2'])
input_y = input_data['open_flag']
train_X, validate_X, train_y, validate_y = train_test_split(input_X, input_y, train_size=0.8, test_size=0.2, random_state=0)

# Defines different columns.
categorical_columns = ['country_code', 'attr_3', 'domain']
numerical_columns   = [col for col in input_X.columns if col not in categorical_columns]

# Defines the feature pre-processing steps.
categorical_transformer = OneHotEncoder(categories='auto', handle_unknown='ignore')
numerical_transformer = Pipeline(steps=[
	('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
	('normalize', Normalizer())])
transformer = ColumnTransformer(transformers=[
	('categorical', categorical_transformer, categorical_columns),
	('numerical',   numerical_transformer,   numerical_columns)])

# Defines the model.
model = RandomForestRegressor(n_estimators=100, n_jobs=2, random_state=0, verbose=2)

# Defines the entire pipeline.
pipeline = Pipeline(steps=[
	('pre-process', transformer),
	('model', model)])

# Tries to train the model.
pipeline.fit(input_X, input_y)

# Tries to score the model.
pipeline.score(validate_X, validate_y)
