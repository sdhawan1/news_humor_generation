# this file should have all the helper functions for the CBOW training & inference script.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# stopwords: global vbl imported from nltk

nltk_sw_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
             'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
             'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
             'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
             'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before',
             'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
             'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
             'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd',
             'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
             'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
             'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

sw = set(nltk_sw_list)

# function to define the model (for ease of use)
class CBOWLanguageModeler(nn.Module):

  def __init__(self, vocab_size, context_size, batch_size=1, embedding_dim=100):
    super(CBOWLanguageModeler, self).__init__()
    self.batch_size = batch_size
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)
    # shortening the below dims to 64, since embedding dim is only 100...
    self.linear1 = nn.Linear(context_size * embedding_dim, 64)
    self.linear2 = nn.Linear(64, vocab_size)

  def forward(self, inputs):
    # create the proper 2D matrix from the embeddings...
    embeds = self.embeddings(inputs).view((self.batch_size, -1))
    out = F.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probs = F.log_softmax(out, dim=1)
    return log_probs



# helper function: return the cbow-styled input to neural net.
# inputs:
#     sentence: input sentence split into tokens.
#     i: index of the target word.
#     window_size: how big is the context window.
# outputs:
#     (returns the key word as well as a vector of tokens, including padding).
def cbow(sentence, i, window_size):
  pad_tok = 0
  iword = sentence[i]
  left = sentence[max(i - window_size, 0):i]
  right = sentence[i+1:i+1+window_size]
  # add the left & right padding to make sure all sizes consistent.
  l_pad = [pad_tok for _ in range(window_size - len(left))]
  r_pad = [pad_tok for _ in range(window_size - len(right))]
  return iword, (l_pad + left + right + r_pad)
