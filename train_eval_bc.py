#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
import pandas as pd
import argparse
from modules.utils import dataholder, dataholder_bert
import json

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("torch version: ",torch.__version__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



parser = argparse.ArgumentParser()

parser.add_argument("-dataset", type = str, help = "select dataset / task", default = "sst")
parser.add_argument("-out_dataset", type = str, help = "select dataset / task", default = "sst")
parser.add_argument("-encoder", type = str, help = "select encoder", default = "lstm", choices = ["lstm", "gru", "mlp", "cnn", "bert"])
parser.add_argument("-data_dir", type = str, help = "directory of saved processed data", default = "data/")
parser.add_argument("-model_dir", type = str, help = "directory to save models", default = "test_models/")
parser.add_argument("-experiments_dir", type = str, help = "directory to save models", default = "test_experiment_results/")
parser.add_argument("-mechanism", type = str, help = "choose mechanism", default = "dot", choices = ["tanh", "dot"] )
parser.add_argument('-operation',  type = str, help='operation over scaled embedding', 
                    default='sum-over', choices = ["sum-over", "max-pool", "mean-pool"])
parser.add_argument('--speed_up', help='can be used to speed up decision flip experiments as mentioned in README', action='store_true')
parser.add_argument('-alpha', type = float, help = 'L2 loss coefficient', default = 20)
parser.add_argument('-perc_inp',type = float, help = 'Input percentage', default = 0.2)
parser.add_argument('-vanilla', help = 'Flag to say if it is vanilla FT or D-Ref', action='store_true')

print("\n", vars(parser.parse_args()), "\n")

args = vars(parser.parse_args())

print("args: ",args)
print("perc_inp: ",args["perc_inp"])

vanilla = args["vanilla"]
print('vanilla: ',vanilla)
dataset = args["dataset"]
out_dataset = args["out_dataset"]
encode_select = args["encoder"]
data_dir= args["data_dir"]
model_dir = args["model_dir"] 
sys.path.append(data_dir)
method = [k for k,v in args.items() if v is True]
print("method: ",method)
save_path = [model_dir + method[0] + "_" + dataset + "/" if len(method) > 0 else model_dir + dataset + "/"][0]

experiments_path = [args["experiments_dir"] + method[0] + "_" + dataset + "/" if len(method) > 0 else args["experiments_dir"] + dataset + "/"][0]

try:

  os.makedirs(save_path + "/model_run_stats/")
  print("\n--Models saved in: {}".format(save_path))

except:

  print("\n--Models saved in: {}".format(save_path))
  
try:

  os.makedirs(experiments_path)
  print("--Experiment results saved in {}".format(experiments_path))

except:

  print("--Experiment results saved in {}".format(experiments_path))

args["bert_model"] = "bert-base-uncased"

if args["dataset"] == "mimicanemia":

  args["bert_model"] = "bert-base-uncased"

#seeds = [24,92, 7, 88, 15]


seeds = [92]

if args["encoder"] == "bert":

    data = dataholder_bert(data_dir, dataset, 8, args["bert_model"]) 
    out_data = dataholder_bert(data_dir, out_dataset, 8, args["bert_model"])
else:

    
    data = dataholder(data_dir, dataset, 32)
    out_data = dataholder(data_dir, out_dataset, 32)

# In[18]:


vocab_size = data.vocab_size
embedding_dim = data.embedding_dim
hidden_dim = data.hidden_dim


tasc_method = method[0] if len(method) > 0 else None
print("tasc_method: ",tasc_method)
hidden_dim = 64 if args["encoder"] != "bert" else 768 // 2
embedding_dim = 300 if args["encoder"] != "bert" else 768
epochs = 20 if args["encoder"] != "bert" else 6 

## special case for twitter without tasc and tanh (not balaned dataset)
if (args["dataset"] == "twitter") & (tasc_method == None):

    epochs = 30

    if args["encoder"] == "mlp":

        epochs = 35


args.update({"vocab_size" : vocab_size, "embedding_dim" : embedding_dim, "hidden_dim" : hidden_dim, 
             "tasc" : tasc_method , "seeds" : seeds, "optimiser":"Adam", "loss":"CrossEntropyLoss",
             "epochs" : epochs, "save_path":save_path, "experiments_path": experiments_path})

print(args)

#### saving config file for this run

with open('modules/config.txt', 'w') as file:
     file.write(json.dumps(args))
     

### training and evaluating models

from modules.run_binary_classification import train_binary_classification_model, evaluate_trained_bc_model
import os 

fname = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + "_predictive_performances.csv"

if os.path.isfile(fname): 

  print(" **** model already exists at {}".format(
    fname
  ))

else:
  print("\nTraining\n")

  train_binary_classification_model(data, out_data, vanilla)
   
  print("\Evaluating\n")

  evaluate_trained_bc_model(data)

  print("Evaluating on out of domain data")

  evaluate_trained_bc_model(out_data)

evaluate_trained_bc_model(data)

print("Evaluating on out of domain data")

evaluate_trained_bc_model(out_data)


### conducting experiments

## special case for mimic

if args["encoder"] == "bert":

    if (args["dataset"] == "mimicanemia" and args["tasc"] is None):

        del data

        data = dataholder_bert(data_dir, dataset, 4, args["bert_model"])

    if (args["dataset"] == "imdb" and args["tasc"] is None):

        del data

        data = dataholder_bert(data_dir, dataset, 4, args["bert_model"])
    
    if (args["dataset"] == "mimicanemia" and args["tasc"] is not None):

        del data

        data = dataholder_bert(data_dir, dataset, 2, args["bert_model"])
    
    if (args["dataset"] == "imdb" and args["tasc"] is not None):

        del data

        data = dataholder_bert(data_dir, dataset, 2, args["bert_model"])


print("\nExperiments\n")




