import torch
import torch.nn
from torch.nn import functional as F

# hyperparameters
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#


torch.manual_seed(1337)

#load tiny shakespeare dataset
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()
    
#get all the unique characters
characters = sorted(list(set(text)))
vocab_size = len(characters)


#map from characters to integers
stoi = {ch:i for i,ch in enumerate(characters)}
itos = {i:ch for i,ch in enumerate(characters)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])