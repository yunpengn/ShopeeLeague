# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import pandas as pd

# Reads the input.
orders = pd.read_csv('orders_sample.csv')

# Defines some useful variables.
result = []
count = 0
print_batch_size = 1000
total_size = len(orders)

# Iterates over each row in the dataframe.
for index, row in orders.iterrows():
  # Prints progess.
  if count % print_batch_size == 0:
  	print('Current progress: %6d / %d' % (count, total_size))

  # Retrieves individual data.
  order_id = row['orderid']
  time_pick = int(row['pick'])
  time_1st = int(row['1st_deliver_attempt'])

  # Appends to result.
  result.append([order_id, 0])

  # Increases the counter.
  count += 1

# Writes the output.
output = pd.DataFrame(result, columns=['orderid', 'is_late'])
output.to_csv('result.csv', index=False)
