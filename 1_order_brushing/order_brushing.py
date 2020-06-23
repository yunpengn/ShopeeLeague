# -*- coding: utf-8 -*-

from datetime import datetime
import numpy as np
import pandas as pd

orders = pd.read_csv('https://gist.githubusercontent.com/yunpengn/7dae8cb6c757a9084c49bce1fec31bab/raw/e185f047d597ff363fab5446e5e13f4546d3e639/order_brushing.csv')
orders.head()

def to_time(event_time):
  return datetime.strptime(event_time, '%Y-%m-%d %H:%M:%S')

def larger_than_hour(event_time1, event_time2):
  time1 = to_time(event_time1)
  time2 = to_time(event_time2)
  return (time2 - time1).total_seconds() > 3600

# Used to save result.
all_shops = []
all_users = []

count = 0
for shop_id, shop_orders in orders.groupby('shopid'):
  # Updates prompt.
  if count % 50 is 0:
    print('#{}'.format(count))

  # Sort by order time.
  shop_orders = shop_orders.sort_values('event_time')
  illegal_orders = None

  # Two pointer-approach.
  slow_pointer = 0
  fast_pointer = 0
  while True:
    if fast_pointer >= len(shop_orders):
      break

    first_row = shop_orders.iloc[slow_pointer]
    last_row  = shop_orders.iloc[fast_pointer]

    # Checks time difference:
    if larger_than_hour(first_row['event_time'], last_row['event_time']):
      slow_pointer += 1
      continue
    
    # Checks whether order brushing has happended
    num_orders = fast_pointer - slow_pointer + 1
    num_buyers = shop_orders.iloc[slow_pointer:fast_pointer + 1]['userid'].nunique()
    if 1.0 * num_orders / num_buyers < 3:
      fast_pointer += 1
      continue

    if illegal_orders is None:
      illegal_orders = shop_orders.iloc[slow_pointer:fast_pointer + 1]
    else:
      illegal_orders = illegal_orders.append(shop_orders.iloc[slow_pointer:fast_pointer + 1])
    
    # Increments fast pointer.
    fast_pointer += 1
  
  # Checks whether this shop has order brushing.
  user_id = "0"
  if illegal_orders is not None:
    buy_times = illegal_orders['userid'].value_counts(sort=True, ascending=False)

    # Finds all needed users.
    users = []
    largest = buy_times.iloc[0]
    for user_id, times in buy_times.iteritems():
      if times != largest:
        break
      
      users.append(str(user_id))

    # Joins together.
    user_id = "&".join(users)
  
  # Saves to result.
  all_shops.append(shop_id)
  all_users.append(user_id)

  # Increments index.
  count += 1

output = pd.DataFrame({'shopid': all_shops, 'userid': all_users})
output.to_csv('result.csv', index=False)
