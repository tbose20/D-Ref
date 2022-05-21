#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch import optim
import json 
import numpy as np
import pandas as pd
from tqdm import tqdm
import confidence_interval

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

if args["mechanism"] == "dot":
    
    from modules.model_components_bc.attentions import DotAttention as attention_mech
    
else:
    
    from modules.model_components_bc.attentions import TanhAttention as attention_mech

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from modules.model_components_bc.encoder import Encoder
from modules.model_components_bc.classifier import train, test
from modules.model_components_bc import tasc

from modules.run_experiments import *
from modules.model_components_bc.classifier import Model


def optimiser_fun(model, encoder):
    
    optimiser = getattr(torch.optim, args["optimiser"]) 

    if args["encoder"] == "bert":
        

        optimiser = optim.AdamW([   
                    {'params': model.encoder.parameters(), 'lr': 1e-5},
                    {'params': model.output.parameters(), 'lr': 1e-5},
                    {'params': model.attention.parameters(), 'lr': 1e-5}
                ], eps = 1e-8, amsgrad = False, weight_decay = 10e-5)




    
    else:
        
        optimiser = optim.Adam([ param for param in model.parameters() if param.requires_grad == True],
                                   amsgrad = True, 
                                   weight_decay = 10e-5)
        
    return optimiser
        
        

def train_binary_classification_model(data,out_data, vanilla):
    
    
    """
    Trains models and monitors on dev set
    Also produces statistics for each run (seed)    
    """
    run_train = 0
    for number in range(len(args["seeds"])):
      best_dev = 0
      torch.manual_seed(args["seeds"][number])
      np.random.seed(args["seeds"][number])
      for ind_al in range(1):

          
        attention = attention_mech(args["hidden_dim"])
                  
        encoder = Encoder(embedding_dim=args["embedding_dim"],
                        vocab_size=data.vocab_size,
                         hidden_dim=args["hidden_dim"], 
                         encode_sel = args["encoder"],
                     embedding = data.pretrained_embeds)
        
            
        tasc_mech = None
            
    
        model = Model(encoder = encoder, 
                      attention = attention, 
                        mask_list=data.mask_list,
                     hidden_dim=args["hidden_dim"],
                     output_dim=data.output_size, tasc = tasc_mech)
        
        model.to(device)
        
        loss_function = nn.CrossEntropyLoss()
        
        if args["encoder"] == "bert":
        
            #model.encoder.bert.embeddings.word_embeddings.weight.requires_grad = False
    
            optimiser = optimiser_fun(model, args["encoder"])
            
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)
    
            print("Total Params:", total_params)
            print("Total Trainable Params:", total_trainable_params)
            
            #assert (total_params - total_trainable_params) == model.encoder.bert.embeddings.word_embeddings.weight.numel()
        else:
            
            model.encoder.embedding.weight.requires_grad = False
            
            optimiser = optimiser_fun(model, args["encoder"])
           
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(p.numel() for p in model.parameters()
                                         if p.requires_grad)
        
            print("Total Params:", total_params)
            print("Total Trainable Params:", total_trainable_params)
            assert (total_params - total_trainable_params) == model.encoder.embedding.weight.numel()
    
        save_folder = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + str(number) + ".model"
  
        ret_model, dev_results, results_to_save = train(model,  
              data.training, 
              data.development, 
              loss_function,
            optimiser,
            epochs = args["epochs"],
              cutoff = False, 
              save_folder = save_folder,
              run = run_train,
              data=data,
              out_data=out_data, vanilla = vanilla)
        temp_res,temp_loss,_,_ = test(ret_model,loss_function,data.development)
        print("Returned model results: ",round(temp_res["macro avg"]["f1-score"],3))


        dev_f1 = round(dev_results["macro avg"]["f1-score"],3)
        if dev_f1 > best_dev:
       #     torch.save(ret_model.state_dict(),save_folder)
            best_dev = dev_f1
            #best_al = al
            text_file = open(args["save_path"]  +"model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_run_" + str(run_train + 1) + ".txt", "w")
            text_file.write(results_to_save)
            text_file.close()
        
        
      run_train +=1
        
        
      print("Best alpha for run "+str(run_train + 1)+": " +" best dev f1: ",str(best_dev)) 
      df = pd.DataFrame(dev_results)
      df.to_csv(args["save_path"]  +"model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_best_model_devrun:" + str(number) + ".csv")
        
import glob
import os 
def evaluate_trained_bc_model(data):    
    
    """
    Runs trained models on test set
    Also keeps the best model for experimentation
    and produces statistics    
    """
    
    saved_models = glob.glob(args["save_path"] + "*.model")
    
    stats_report = {}
 
    stats_report[args["mechanism"]] = {}
                
    Actual = []
    Predicted = []

    for j in range(len(args["seeds"])):
        
        torch.manual_seed(args["seeds"][j])
        np.random.seed(args["seeds"][j])
  
          
        attention = attention_mech(args["hidden_dim"])
                  
        encoder = Encoder(embedding_dim=args["embedding_dim"],
                        vocab_size=data.vocab_size,
                         hidden_dim=args["hidden_dim"], 
                         encode_sel = args["encoder"],
                     embedding = data.pretrained_embeds)
        
            
            
        tasc_mech = None
            
    
        model = Model(encoder = encoder, 
                      attention = attention, 
                        mask_list=data.mask_list,
                     hidden_dim=args["hidden_dim"],
                     output_dim=data.output_size, 
                     tasc = tasc_mech)
        
        model.to(device)
        
        
        current_model = args["save_path"] + args["encoder"] + "_" + args["mechanism"] + str(j) + ".model"
        
        print("current_model: ",current_model)

        index_model = saved_models.index(current_model)
        
        # loading the trained model
    
        model.load_state_dict(torch.load(saved_models[index_model], map_location=device))
        
        model.to(device)
        
        loss_function = nn.CrossEntropyLoss()

        test_results,test_loss,actual,predicted = test(model, loss_function, data.testing)
        
        Actual.extend(actual)
        Predicted.extend(predicted)
       
        df = pd.DataFrame(test_results)
       
        df.to_csv(args["save_path"]  +"/model_run_stats/" + args["encoder"] + "_" + args["mechanism"] + "_best_model_testrun:" + str(j) + ".csv")
     
        stats_report[args["mechanism"]]["Macro F1 - avg:run:" +str(j)] = test_results["macro avg"]["f1-score"]
        
             
        print("Run: ", j, 
              " Test loss: ", round(test_loss), 
              " Test accuracy: ", round(test_results["macro avg"]["f1-score"], 3),
             )
    
    confidence_interval.confidence_interval(Actual,Predicted)
 
    
    """
    now to keep only the best model
    
    """
    
 #   performance_list = tuple(stats_report[args["mechanism"]].items()) ## keeping the runs and acuracies
 #   
 #   performance_list = [(x.split(":")[-1], y) for (x,y) in performance_list]
 #   
 #   sorted_list = sorted(performance_list, key = lambda x: x[1])
 #   
 #   models_to_get_ridoff, _ = zip(*sorted_list[:len(args["seeds"]) - 1])
 #   
 #   for item in models_to_get_ridoff:
 #       
 #       os.remove(args["save_path"] + args["encoder"] + "_" + args["mechanism"]  + str(item) + ".model")
    
    """
    saving the stats
    """
    
    stats_report[args["mechanism"]]["mean"] = np.asarray(list(stats_report[args["mechanism"]].values())).mean()
    stats_report[args["mechanism"]]["std"] = np.asarray(list(stats_report[args["mechanism"]].values())).std()
    
    df = pd.DataFrame(stats_report)
    df.to_csv(args["save_path"] + args["encoder"] + "_" + args["mechanism"] + "_predictive_performances.csv")
    


