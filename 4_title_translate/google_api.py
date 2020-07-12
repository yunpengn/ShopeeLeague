# -*- coding: utf-8 -*-

from googletrans import Translator
import math
import numpy as np
import pandas as pd

# Reads the input.
products = pd.read_csv('test_tcn.csv')

# Defines the translator.
translator = Translator()

# Defines some useful variables.
result = []
count = 0
print_batch_size = 100
total_size = len(orders)

# Iterates over each row in the dataframe.
for index, row in orders.iterrows():
  # Prints progess.
  if count % print_batch_size == 0:
    print('Current progress: %4d / %d' % (count, total_size))

  # Translates the product title.
  title_input = row['text']
  title_output = translator.translate(title_input, src='zh-tw', dest='en')

# Writes the output.
output = pd.DataFrame(result, columns=['translation_output'])
output.to_csv('result.csv', index=False)
