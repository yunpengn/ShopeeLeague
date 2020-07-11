# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import pandas as pd

# Reads the input.
orders = pd.read_csv('orders_sample.csv')

# Defines the SLA matrix.
time_sla = {
  'Manila':   {'Manila': 3, 'Luzon': 5, 'Visayas': 7, 'Mindanao': 7},
  'Luzon':    {'Manila': 5, 'Luzon': 5, 'Visayas': 7, 'Mindanao': 7},
  'Visayas':  {'Manila': 7, 'Luzon': 7, 'Visayas': 7, 'Mindanao': 7},
  'Mindanao': {'Manila': 7, 'Luzon': 7, 'Visayas': 7, 'Mindanao': 7}
}

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

  # Retrieves time-based data.
  order_id    = row['orderid']
  time_pick   = int(row['pick'])
  time_1st    = int(row['1st_deliver_attempt'])
  time_2nd    = int(row['2nd_deliver_attempt'])

  # Retrieves standard delivery SLA.
  city_buyer  = row['buyeraddress'].split(' ')[-1]
  city_seller = row['selleraddress'].split(' ')[-1]
  sla_1st     = time_sla[city_buyer][city_seller]
  sla_2nd     = 3

  # Checks whether the delivery is late.
  is_late = False
  if num_working_dayas(time_pick, time_1st) > sla_1st:
    is_late = True
  elif num_working_dayas(time_1st, time_2nd) > sla_2nd:
    is_late = True

  # Appends to result.
  result.append([order_id, int(is_late)])

  # Increases the counter.
  count += 1

# Writes the output.
output = pd.DataFrame(result, columns=['orderid', 'is_late'])
output.to_csv('result.csv', index=False)
