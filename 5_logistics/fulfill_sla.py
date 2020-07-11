# -*- coding: utf-8 -*-

from datetime import date
from datetime import datetime
from datetime import timedelta
import math
import numpy as np
import pandas as pd

# Reads the input.
orders = pd.read_csv('orders.csv')

# Defines the SLA matrix.
time_sla = {
  'manila':   {'manila': 3, 'luzon': 5, 'visayas': 7, 'mindanao': 7},
  'luzon':    {'manila': 5, 'luzon': 5, 'visayas': 7, 'mindanao': 7},
  'visayas':  {'manila': 7, 'luzon': 7, 'visayas': 7, 'mindanao': 7},
  'mindanao': {'manila': 7, 'luzon': 7, 'visayas': 7, 'mindanao': 7}
}

# Defines some datetime units.
days = {
  'mon': 0,
  'tue': 1,
  'wed': 2,
  'thu': 3,
  'fri': 4,
  'sat': 5,
  'sun': 6
}
delta_day = timedelta(days=1)
h1 = datetime.strptime('2020-03-08', '%Y-%m-%d').date()
h2 = datetime.strptime('2020-03-25', '%Y-%m-%d').date()
h3 = datetime.strptime('2020-03-30', '%Y-%m-%d').date()
h4 = datetime.strptime('2020-03-31', '%Y-%m-%d').date()
hard_limit = 15

# Defines how to calculate # of working days.
def num_working_days(start_timestamp, end_timestamp):
  date_start = datetime.fromtimestamp(start_timestamp).date()
  date_end   = datetime.fromtimestamp(end_timestamp).date()

  dt = date_start + delta_day
  day_count = 0
  while dt <= date_end and day_count < hard_limit:
    # Counts the number of working days.
    if dt.weekday() != days['sun'] and dt not in [h1, h2, h3, h4]:
      day_count += 1
    
    dt += delta_day

  return day_count

# Defines some useful variables.
result = []
count = 0
print_batch_size = 1000
total_size = len(orders)

# Iterates over each row in the dataframe.
for index, row in orders.iterrows():
  # Prints progess.
  if count % print_batch_size == 0:
    print('Current progress: %7d / %d' % (count, total_size))

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
