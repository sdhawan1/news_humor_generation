## let's add the preprocessing functionality here.

import csv
import os
import numpy as np
import pandas as pd

def get_label_bin(score):
  # if score < 0.5:
  #   return 0
  # elif score >= 0.5 and score < 1.5:
  #   return 1
  # elif score >= 1.5 and score < 2.5:
  #   return 2
  # else:
  #   return 3
  if score < 1:
    return 0
  else:
    return 1

#later, write a method to preprocess that works better for other tasks.

#in this function, preprocess the data
def preprocess_data(filename, label_method = 'binary', MAX_SEQ_LENGTH = 128):
  f = open(filename, "r")
  reader = csv.reader(f)
  next(reader)
  data = []
  funny, not_funny = 0, 0
  for row in reader:
    sentence = row[1]
    #print(sentence)
    replacement = row[2]
    #print(replacement)
    score = float(row[-1])
    if label_method == 'binary':
      label = get_label_bin(score)
    if not label:
      not_funny += 1
    else:
      funny += 1
    old_word = ""
    flag = False
    for c in sentence:
      if c == "<" or flag:
        old_word += c
        flag = True
      if c == ">":
        flag = False
    #print(old_word)
    new_sentence = sentence.replace(old_word, replacement)
    if len(new_sentence) > MAX_SEQ_LENGTH:
      MAX_SEQ_LENGTH = len(new_sentence)
    #print(new_sentence)
    data.append([new_sentence, label])
  print(len(data))
  return data, funny, not_funny
