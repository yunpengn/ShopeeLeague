# -*- coding: utf-8 -*-

from datetime import datetime
import math
import numpy as np
import pandas as pd

# Reads the input.
orders = pd.read_csv('orders_sample.csv')

# Defines the SLA matrix.
time_sla = {
  'manila':   {'manila': 3, 'luzon': 5, 'visayas': 7, 'mindanao': 7},
  'luzon':    {'manila': 5, 'luzon': 5, 'visayas': 7, 'mindanao': 7},
  'visayas':  {'manila': 7, 'luzon': 7, 'visayas': 7, 'mindanao': 7},
  'mindanao': {'manila': 7, 'luzon': 7, 'visayas': 7, 'mindanao': 7}
}

# Defines some useful variables.
result = []
count = 0
print_batch_size = 1000
total_size = len(orders)

# Defines how to calculate # of working days.
def num_working_days(start_timestamp, end_timestamp):
  return 0

# Iterates over each row in the dataframe.
for index, row in orders.iterrows():
  # Prints progess.
  if count % print_batch_size == 0:
    print('Current progress: %6d / %d' % (count, total_size))

  # Retrieves time-based data.
  order_id    = row['orderid']
  time_pick   = int(row['pick'])
  time_1st    = int(row['1st_deliver_attempt'])
  time_2nd    = row['2nd_deliver_attempt']

  # Retrieves standard delivery SLA.
  city_buyer  = row['buyeraddress'].split(' ')[-1].lower()
  city_seller = row['selleraddress'].split(' ')[-1].lower()
  sla_1st     = time_sla[city_buyer][city_seller]
  sla_2nd     = 3

  # Checks whether the delivery is late.
  is_late = False
  if num_working_days(time_pick, time_1st) > sla_1st:
    is_late = True
  elif not math.isnan(time_2nd) and num_working_days(time_1st, int(time_2nd)) > sla_2nd:
    is_late = True

  # Appends to result.
  result.append([order_id, int(is_late)])

  # Increases the counter.
  count += 1

# Writes the output.
output = pd.DataFrame(result, columns=['orderid', 'is_late'])
output.to_csv('result.csv', index=False)
