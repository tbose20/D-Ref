import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.autograd import Variable
import pandas as pd
from modules.experiments_bc.eval_utils import * 
from collections import OrderedDict
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from tqdm import trange
import json
import math
from collections import OrderedDict
import sys

sys.path.insert(1,'captum/')

from captum.attr import DeepLift

"""MODEL OUT DISTRIBUTIONS"""

with open('modules/config.txt', 'r') as f:
    args = json.load(f)


def stat_filter(data,model,dataset,glob_list, glob_nhate_list):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    torch.cuda.manual_seed_all(100)
    #torch.use_deterministic_algorithms(True)


    filt_tokens_FP = []
    filt_tokens_TP = []
    sent_FP = []

    filt_tokens_FN = []
    filt_tokens_TN = []
    sent_FN = []

    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.requires_grad_(True)

    else:
        model.encoder.embedding.weight.requires_grad_(True)
    
    dl = DeepLift(model)

    for sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        # original trained model    
        model.train()
        
        inpt_seq = sentences.long() 

        model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)


        yhat = model(model.encoder.embed,sentences, lengths, retain_gradient = True)
        masks = model.masks       
        em = model.encoder.embed
        yhat.max(-1)[0].sum().backward() 

        pred_lab = yhat.max(-1).indices.cpu().numpy()

        
        dl_attr = dl.attribute(model.encoder.embed, additional_forward_args = sentences, target= yhat.max(-1).indices)
        
        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]

        with torch.no_grad():

            
            att_source_set = {}
            rand_set = {}
            att_grad_set = {}
            att_mul_grad_set = {}

            grad_set = {}
            ig_set = {}
            omission_set = {}
            
            sent_arr = [np.array(s) for s in sent_tok]
            dl_attr = dl_attr.sum(-1)[:,:max(lengths)]

            dl_attr.masked_fill_(masks[:,:max(lengths)].bool(), float("-inf"))
    

            top_DL = torch.topk(dl_attr, k = dl_attr.size(1))[1].to(device)

            
            extr_tok_dl = [list(sent_arr[ind][top_DL.cpu()[ind]]) for ind in range(len(sent_arr))]

            top_inst_tok = [ext[:math.floor(args["perc_inp"]*len(ext))] for ext in extr_tok_dl]

            lab = labels.cpu().numpy()
            for inst in range(len(pred_lab)):
                if pred_lab[inst] == 1 and lab[inst] == 0:
                    sent_FP.append(list(sent_arr[inst][top_DL.cpu()[inst]])) #getting a ranked list of tokens for FP sentences in dev
                    filt_tokens_FP.extend(list(set(top_inst_tok[inst]) & set(glob_list)))
            for inst in range(len(pred_lab)):
                if pred_lab[inst] == 1 and lab[inst] == 1:
                    filt_tokens_TP.extend(list(set(top_inst_tok[inst])))
            
            for inst in range(len(pred_lab)):
                if pred_lab[inst] == 0 and lab[inst] == 1:
                    sent_FN.append(list(sent_arr[inst][top_DL.cpu()[inst]])) #getting a ranked list of tokens for FP sentences in dev
                    filt_tokens_FN.extend(list(set(top_inst_tok[inst]) & set(glob_nhate_list)))
            for inst in range(len(pred_lab)):
                if pred_lab[inst] == 0 and lab[inst] == 0:
                    filt_tokens_TN.extend(list(set(top_inst_tok[inst])))




            filt_tok_FP_only = [tok for tok in filt_tokens_FP if tok not in filt_tokens_TP]
            filt_tok_TP_only = [tok for tok in filt_tokens_TP if tok not in filt_tokens_FP]
            

            filt_tok_FN_only = [tok for tok in filt_tokens_FN if tok not in filt_tokens_TN]
            filt_tok_TN_only = [tok for tok in filt_tokens_TN if tok not in filt_tokens_FN]


    return filt_tok_FP_only, filt_tok_TP_only, sent_FP,  filt_tok_FN_only, filt_tok_TN_only, sent_FN




def extract_ftr_attr(data, out_data, model, epoch, dataset,dataset_name, out_dataset_name):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    torch.cuda.manual_seed_all(100)
    #torch.use_deterministic_algorithms(True)

    dl = DeepLift(model)

    pbar = trange(len(data) * data.batch_size, desc='set of tokens', leave=True)

    
    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.requires_grad_(True)

    else:
        model.encoder.embedding.weight.requires_grad_(True)

    all_sent = []
    all_sent_decode = []
    all_imp_tok_DL = []
    pred_lab = []
    all_dl_inst_sc = []
    top_dl_scores = []



    inv_doc_freq = dataset.inv_doc_freq_dict
    inv_doc_freq_hate = dataset.inv_doc_freq_dict_hate
    inv_doc_freq_nhate = dataset.inv_doc_freq_dict_nhate
    od = OrderedDict()

    thresh_df = 0.95*max(inv_doc_freq.values())
    thresh_df_hate = 0.95*max(inv_doc_freq_hate.values())
    thresh_df_nhate = 0.95*max(inv_doc_freq_nhate.values())

    thresh_df_min = 0.09

    #vocab = dataset.tokenizer.vocab
    attr_voc_sc = np.zeros((dataset.tokenizer.vocab_size,6)) #For storing average of token-wise global scores for each attr metric
    attr_voc_sc_nhate = np.zeros((dataset.tokenizer.vocab_size,6))
    attr_voc_sc_hate = np.zeros((dataset.tokenizer.vocab_size,6))
    attr_voc_sc2 = np.zeros((dataset.tokenizer.vocab_size,6)) #For storing average of token-wise global scores for each attr metric
    attr_voc_sc2_nhate = np.zeros((dataset.tokenizer.vocab_size,6))
    attr_voc_sc2_hate = np.zeros((dataset.tokenizer.vocab_size,6))
 
    count_voc_sc = np.zeros((dataset.tokenizer.vocab_size,6)) #For maintaining count required for avg calculation
    count_voc_sc_nhate = np.zeros((dataset.tokenizer.vocab_size,6))
    count_voc_sc_hate = np.zeros((dataset.tokenizer.vocab_size,6))

    mult_invd = [inv_doc_freq[ind] if ind in list(inv_doc_freq.keys()) else 0 for ind in list(range(count_voc_sc.shape[0]))]
    mult_invd_hate = [inv_doc_freq_hate[ind] if ind in list(inv_doc_freq_hate.keys()) else 0 for ind in list(range(count_voc_sc_hate.shape[0]))]
    mult_invd_nhate = [inv_doc_freq_nhate[ind] if ind in list(inv_doc_freq_nhate.keys()) else 0 for ind in list(range(count_voc_sc_nhate.shape[0]))]

    keys = list(inv_doc_freq.keys())
    keys_hate = list(inv_doc_freq_hate.keys())
    keys_nhate = list(inv_doc_freq_nhate.keys())



    mult_invd = [1 if mi < thresh_df and mi > thresh_df_min else 0 for mi in mult_invd]
    mult_invd_hate = [1 if mi < thresh_df_hate and mi > thresh_df_min else 0 for mi in mult_invd_hate]
    mult_invd_nhate = [1 if mi < thresh_df_nhate and mi > thresh_df_min else 0 for mi in mult_invd_nhate]


    mult_invd = np.array(mult_invd)
    mult_invd_hate = np.array(mult_invd_hate)
    mult_invd_nhate = np.array(mult_invd_nhate)



    for sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        # original trained model    
        model.train()
       
   
        inpt_seq = sentences.long() 

        model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)


        yhat = model(model.encoder.embed, sentences, lengths, retain_gradient = True)
        masks = model.masks       
        em = model.encoder.embed

        yhat.max(-1)[0].sum().backward()

        yhat_lab = yhat.max(-1).indices


        dl_grads = dl.attribute(model.encoder.embed, additional_forward_args = sentences, target= yhat_lab)
        dl_grads = dl_grads.sum(-1)[:,:max(lengths)]


        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]
        all_sent.extend(sent_tok)
        all_sent_decode.extend(dataset.tokenizer.batch_decode(sentences))

        with torch.no_grad():
            
            dl_grads1 = dl_grads.detach().clone()
            dl_grads.masked_fill_(masks[:,:max(lengths)].bool(), float("-inf"))
            dl_grads[dl_grads != dl_grads] = -9999
            bool_val = ~torch.isinf(dl_grads)
           
            for ind in range(dl_grads.size(0)):
                  valid_values = torch.masked_select(dl_grads[ind,:],bool_val[ind,:])

                  if ind == 0:
                      if valid_values.size(0) == 1:
                          min_max_row = dl_grads1[ind,:] - valid_values.item() #For a single word sentence, sigmoid will yield 0.5 as score is 0
                          min_max_row = torch.reshape(min_max_row,(1,min_max_row.size(0)))
                          DL_min_max = min_max_row
 
                      else:
                          min_row = torch.min(valid_values)
                          max_row = torch.max(valid_values)

                          minmaxr = max_row-min_row
                          min_max_row = torch.div(torch.sub(dl_grads1[ind,:],min_row.item()),minmaxr.item())
                          min_max_row = torch.reshape(min_max_row,(1,min_max_row.size(0)))
                          DL_min_max = min_max_row


                  else:
                      if len(valid_values) == 1:
                          min_max_row = dl_grads1[ind,:] - valid_values.item() #For a single word sentence, sigmoid will yield 0.5 as score is 0

                          min_max_row = torch.reshape(min_max_row,(1,min_max_row.size(0)))

                          DL_min_max = torch.cat((DL_min_max,min_max_row),0)

                      else:
                          min_row = torch.min(valid_values)
                          max_row = torch.max(valid_values)

                          minmaxr = max_row-min_row
                          min_max_row = torch.div(torch.sub(dl_grads1[ind,:],min_row.item()),minmaxr.item())
                          min_max_row = torch.reshape(min_max_row,(1,min_max_row.size(0)))
                          DL_min_max = torch.cat((DL_min_max,min_max_row),0)
                     
            min_max_val = DL_min_max
            DL_min_max = torch.sigmoid(DL_min_max)

            dl_grads1.masked_fill_(masks[:,:max(lengths)].bool(), float("-inf"))  

            dl_grads1[dl_grads1 != dl_grads1] = -9999 #If there are nans

 


            dl_grads = DL_min_max

           
            dl_grads.masked_fill_(masks[:,:max(lengths)].bool(), float("-inf")) 

            dl_grads[dl_grads != dl_grads] = -9999 #If there are nans




            for ind in range(len(dl_grads)):
              for it in range(len(dl_grads[ind])):
                if not math.isinf(dl_grads[ind][it]):
                    attr_voc_sc[sentences[ind][it],2]+=dl_grads[ind][it]
                    attr_voc_sc2[sentences[ind][it],2]+=dl_grads1[ind][it]

                    count_voc_sc[sentences[ind][it],2]+=1
                    #if yhat_lab[ind] == 0:
                    if yhat_lab.dim() != 0:
                        if yhat_lab[ind] == 0:
                            attr_voc_sc_nhate[sentences[ind][it],2]+=dl_grads[ind][it]
                            attr_voc_sc2_nhate[sentences[ind][it],2]+=dl_grads1[ind][it]

                            count_voc_sc_nhate[sentences[ind][it],2]+=1
                        else:
                            attr_voc_sc_hate[sentences[ind][it],2]+=dl_grads[ind][it]
                            attr_voc_sc2_hate[sentences[ind][it],2]+=dl_grads1[ind][it]

                            count_voc_sc_hate[sentences[ind][it],2]+=1
                    else:
                        if yhat_lab == 0:
                            attr_voc_sc_nhate[sentences[0][it],2]+=dl_grads[ind][it]
                            attr_voc_sc2_nhate[sentences[0][it],2]+=dl_grads1[ind][it]

                            count_voc_sc_nhate[sentences[0][it],2]+=1
                        else:
                            attr_voc_sc_hate[sentences[0][it],2]+=dl_grads[ind][it]
                            attr_voc_sc2_hate[sentences[0][it],2]+=dl_grads1[ind][it]

                            count_voc_sc_hate[sentences[0][it],2]+=1


            top_DL = torch.topk(dl_grads1, k = dl_grads1.size(1))[1].to(device) 
            top_DL_sc = torch.topk(dl_grads, k = dl_grads.size(1))[0].to(device)
            top_DL_scaled = torch.topk(dl_grads, k = dl_grads.size(1))[1].to(device)  
          

            all_dl_inst_sc.extend(dl_grads.cpu().numpy().tolist())
            top_dl_scores.extend(top_DL_sc.cpu().numpy().tolist())
             
            ind = 0 
            extr_tok_dl = []

            sent_arr = [np.array(s) for s in sent_tok]
            extr_tok_dl = [list(sent_arr[ind][top_DL.cpu()[ind]]) for ind in range(len(sent_arr))]

            all_imp_tok_DL.extend(extr_tok_dl)
            
            temp = 0
            
            model.eval()

            maximum = max(lengths)
            increments =  torch.round(maximum.float() * 0.02).int()
            
            ## to speed up experiments if you want
            if args["speed_up"]:
            
                increments = max(1,increments)

            else:

                increments = 1


    
        pbar.update(data.batch_size)
        pbar.refresh()

    
    """Saving percentage decision flip"""
    all_imp_tok_DL = [" ".join(s) for s in all_imp_tok_DL]

    count_voc_sc = np.where(count_voc_sc == 0, 1, count_voc_sc)
    
    
    attr_voc_sc = np.divide(attr_voc_sc,count_voc_sc)
    attr_voc_sc2 = np.divide(attr_voc_sc2,count_voc_sc)

    attr_voc_sc = attr_voc_sc*mult_invd[:,None] 
    attr_voc_sc_sort = np.argsort(attr_voc_sc,axis=0)
    attr_voc_sc2_sort = np.argsort(attr_voc_sc2,axis=0)

    attr_voc_sc_sort = attr_voc_sc_sort[::-1,:] #reverse to sort in descending order
    attr_voc_sc2_sort = attr_voc_sc2_sort[::-1,:]
    attr_scores_sort = np.sort(attr_voc_sc,axis=0)
    attr_scores_sort2 = np.sort(attr_voc_sc2,axis=0)

    attr_scores_sort = attr_scores_sort[::-1,:]
    attr_scores_sort2 = attr_scores_sort2[::-1,:]


    tok_voc_sc_sort = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort = [[tok_voc_sc_sort[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort[0]))] 

   
    count_voc_sc_nhate = np.where(count_voc_sc_nhate == 0, 1, count_voc_sc_nhate)
    attr_voc_sc_nhate = np.divide(attr_voc_sc_nhate,count_voc_sc_nhate)
    attr_voc_sc2_nhate = np.divide(attr_voc_sc2_nhate,count_voc_sc_nhate)

    attr_voc_sc_nhate = attr_voc_sc_nhate*mult_invd_nhate[:,None]
    attr_voc_sc_sort_nhate = np.argsort(attr_voc_sc_nhate,axis=0)
    attr_voc_sc_sort_nhate = attr_voc_sc_sort_nhate[::-1,:] #reverse to sort in descending order
    attr_scores_sort_nhate = np.sort(attr_voc_sc_nhate,axis=0)
    attr_scores_sort_nhate = attr_scores_sort_nhate[::-1,:]
    attr_scores_sort2_nhate = np.sort(attr_voc_sc2_nhate,axis=0)
    attr_scores_sort2_nhate = attr_scores_sort2_nhate[::-1,:]


    tok_voc_sc_sort_nhate = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort_nhate[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort_nhate = [[tok_voc_sc_sort_nhate[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort_nhate[0]))]

    count_voc_sc_hate = np.where(count_voc_sc_hate == 0, 1, count_voc_sc_hate)
    count_voc_sc_hate = count_voc_sc_hate

    attr_voc_sc_hate = np.divide(attr_voc_sc_hate,count_voc_sc_hate)
    attr_voc_sc2_hate = np.divide(attr_voc_sc2_hate,count_voc_sc_hate)

    attr_voc_sc_hate = attr_voc_sc_hate*mult_invd_hate[:,None]
    attr_voc_sc_sort_hate = np.argsort(attr_voc_sc_hate,axis=0)
    attr_voc_sc_sort_hate = attr_voc_sc_sort_hate[::-1,:] #reverse to sort in descending order
    attr_scores_sort_hate = np.sort(attr_voc_sc_hate,axis=0)
    attr_scores_sort_hate = attr_scores_sort_hate[::-1,:]
    attr_scores_sort2_hate = np.sort(attr_voc_sc2_hate,axis=0)
    attr_scores_sort2_hate = attr_scores_sort2_hate[::-1,:]


    tok_voc_sc_sort_hate = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort_hate[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort_hate = [[tok_voc_sc_sort_hate[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort_hate[0]))]

 

    return tok_col_voc_sort,tok_col_voc_sort_nhate,tok_col_voc_sort_hate,all_imp_tok_DL 



def percentage_removed(data, out_data, model, save_path,dataset, dataset_name, out_dataset_name):
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    
    pbar = trange(len(data) * data.batch_size, desc='set of tokens', leave=True)

    results_flip = {}
    results_flip["max_source"] = []
    results_flip["random"] = []
    results_flip["lengths"] = []
    results_flip["att_grad"] = []
    results_flip["att*grad"] = []

    results_flip["grad"] = []
    results_flip["omission"] = []
    results_flip["IG"] = []
    
    
    if args["encoder"] == "bert":
        model.encoder.bert.embeddings.requires_grad_(True)

    else:
        model.encoder.embedding.weight.requires_grad_(True)

    all_sent = []
    all_sent_decode = []
    all_imp_tok_IG = []
    all_imp_tok_att = []
    all_imp_tok_attg = []
    all_imp_tok_omm = []
    all_imp_tok_attmulg = []   
    all_imp_tok_grad = [] 
    pred_lab = []

    #vocab = dataset.tokenizer.vocab
    attr_voc_sc = np.zeros((dataset.tokenizer.vocab_size,6)) #For storing average of token-wise global scores for each attr metric
    attr_voc_sc_nhate = np.zeros((dataset.tokenizer.vocab_size,6))
    attr_voc_sc_hate = np.zeros((dataset.tokenizer.vocab_size,6))
    count_voc_sc = np.zeros((dataset.tokenizer.vocab_size,6)) #For maintaining count required for avg calculation
    count_voc_sc_nhate = np.zeros((dataset.tokenizer.vocab_size,6))
    count_voc_sc_hate = np.zeros((dataset.tokenizer.vocab_size,6))
    for sentences, lengths, labels in data:
        
        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":
            
            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
        else:
            
            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        # original trained model    
        model.train()
       
        inpt_seq = sentences.long() 

        model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)

        yhat, weights_or = model(model.encoder.embed, sentences, lengths, retain_gradient = True)
        
        yhat.max(-1)[0].sum().backward(retain_graph = True)
        
       # print("yhat.max(-1):",yhat.max(-1))
       # print("yhat.max(-1).indices:",yhat.max(-1).indices)

        pred_lab.extend(yhat.max(-1).indices.cpu().numpy()) #Uncomment from here ######################
        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)]

        integrated_grads = model.integrated_grads(sentences, 
                                                g, lengths, 
                                                original_pred = yhat.max(-1))
         

        weights_def_grad = model.weights.grad
        
        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]
        all_sent.extend(sent_tok)
        all_sent_decode.extend(dataset.tokenizer.batch_decode(sentences))

        with torch.no_grad():

            att_source_set = {}
            rand_set = {}
            att_grad_set = {}
            att_mul_grad_set = {}

            grad_set = {}
            ig_set = {}
            omission_set = {}
            
            g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            for ind in range(len(g1)):
              for it in range(len(g1[ind])):
                if not math.isinf(g1[ind][it]):
                    attr_voc_sc[sentences[ind][it],0]+=g1[ind][it]
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],0]+=g1[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],0]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],0]+=g1[ind][it]
                      count_voc_sc_hate[sentences[ind][it],0]+=1
                    count_voc_sc[sentences[ind][it],0]+=1
            

            top_grad = torch.topk(g1, k = g1.size(1))[1].to(device)


            omission_scores = model.get_omission_scores(sentences, lengths, yhat)
            for ind in range(len(omission_scores)):
              for it in range(len(omission_scores[ind])):
                if not math.isinf(omission_scores[ind][it]):
                    attr_voc_sc[sentences[ind][it],1]+=omission_scores[ind][it]
                    count_voc_sc[sentences[ind][it],1]+=1
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],1]+=omission_scores[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],1]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],1]+=omission_scores[ind][it]
                      count_voc_sc_hate[sentences[ind][it],1]+=1


            top_omission = torch.topk(omission_scores, k = weights_or.size(1))[1].to(device)

            integrated_grads.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            for ind in range(len(integrated_grads)):
              for it in range(len(integrated_grads[ind])):
                if not math.isinf(integrated_grads[ind][it]):
                    attr_voc_sc[sentences[ind][it],2]+=integrated_grads[ind][it]
                    count_voc_sc[sentences[ind][it],2]+=1
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],2]+=integrated_grads[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],2]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],2]+=integrated_grads[ind][it]
                      count_voc_sc_hate[sentences[ind][it],2]+=1


            top_IG = torch.topk(integrated_grads, k = integrated_grads.size(1))[1].to(device)
           
            for ind in range(len(weights_or)):
              for it in range(len(weights_or[ind])):
                if not math.isinf(weights_or[ind][it]):
                    attr_voc_sc[sentences[ind][it],3]+=weights_or[ind][it]
                    count_voc_sc[sentences[ind][it],3]+=1
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],3]+=weights_or[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],3]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],3]+=weights_or[ind][it]
                      count_voc_sc_hate[sentences[ind][it],3]+=1

 
            top_att = torch.topk(weights_or, k = weights_or.size(1))[1].to(device)
            top_randi = torch.randn(weights_or.shape)
            top_rand = torch.topk(top_randi, k = weights_or.size(1))[1].to(device)

            weight_mul_grad = weights_or * weights_def_grad

            weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            for ind in range(len(weights_def_grad)):
              for it in range(len(weights_def_grad[ind])):
                if not math.isinf(weights_def_grad[ind][it]):
                    attr_voc_sc[sentences[ind][it],4]+=weights_def_grad[ind][it]
                    count_voc_sc[sentences[ind][it],4]+=1
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],4]+=weights_def_grad[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],4]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],4]+=weights_def_grad[ind][it]
                      count_voc_sc_hate[sentences[ind][it],4]+=1


            top_att_grad = torch.topk(weights_def_grad, k = weights_or.size(1))[1].to(device)


            weight_mul_grad.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))
            for ind in range(len(weight_mul_grad)):
              for it in range(len(weight_mul_grad[ind])):
                if not math.isinf(weight_mul_grad[ind][it]):
                    attr_voc_sc[sentences[ind][it],5]+=weight_mul_grad[ind][it]
                    count_voc_sc[sentences[ind][it],5]+=1
                    if labels[ind] == 0:
                      attr_voc_sc_nhate[sentences[ind][it],5]+=weight_mul_grad[ind][it]
                      count_voc_sc_nhate[sentences[ind][it],5]+=1
                    else:
                      attr_voc_sc_hate[sentences[ind][it],5]+=weight_mul_grad[ind][it]
                      count_voc_sc_hate[sentences[ind][it],5]+=1


            top_att_mul_grad = torch.topk(weight_mul_grad, k = weights_or.size(1))[1].to(device)

             
            ind = 0 
            extr_tok_ig = []
            extr_tok_att = []
            extr_tok_attg = []
            extr_tok_grad = []
            extr_tok_grad = []
            extr_tok_omm = []
            extr_tok_attmulg = []


            sent_arr = [np.array(s) for s in sent_tok]
            extr_tok_ig = [list(sent_arr[ind][top_IG.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_att = [list(sent_arr[ind][top_att.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_attg = [list(sent_arr[ind][top_att_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_grad = [list(sent_arr[ind][top_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_omm = [list(sent_arr[ind][top_omission.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_attmulg = [list(sent_arr[ind][top_att_mul_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_ig = [list(sent_arr[ind][top_IG.cpu()[ind]]) for ind in range(len(sent_arr))]



            all_imp_tok_IG.extend(extr_tok_ig)
            all_imp_tok_att.extend(extr_tok_att)
            all_imp_tok_attg.extend(extr_tok_attg)
            all_imp_tok_grad.extend(extr_tok_grad)
            all_imp_tok_omm.extend(extr_tok_omm)
            all_imp_tok_attmulg.extend(extr_tok_attmulg)

            
            temp = 0
            
            model.eval()

            maximum = max(lengths)
            increments =  torch.round(maximum.float() * 0.02).int()
            
            ## to speed up experiments if you want
            if args["speed_up"]:
            
                increments = max(1,increments)

            else:

                increments = 1
    
        pbar.update(data.batch_size)
        pbar.refresh()

    
    """Saving percentage decision flip"""
    all_imp_tok_IG = [" ".join(s) for s in all_imp_tok_IG]
    all_imp_tok_att = [" ".join(s) for s in all_imp_tok_att]
    all_imp_tok_attg = [" ".join(s) for s in all_imp_tok_attg]
    all_imp_tok_grad = [" ".join(s) for s in all_imp_tok_grad]
    all_imp_tok_omm = [" ".join(s) for s in all_imp_tok_omm]
    all_imp_tok_attmulg = [" ".join(s) for s in all_imp_tok_attmulg]

    count_voc_sc = np.where(count_voc_sc == 0, 1, count_voc_sc)
    attr_voc_sc = np.divide(attr_voc_sc,count_voc_sc)
    attr_voc_sc_sort = np.argsort(attr_voc_sc,axis=0)
    attr_voc_sc_sort = attr_voc_sc_sort[::-1] #reverse to sort in descending order
    tok_voc_sc_sort = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort = [[tok_voc_sc_sort[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort[0]))] 
   
    count_voc_sc_nhate = np.where(count_voc_sc_nhate == 0, 1, count_voc_sc_nhate)
    attr_voc_sc = np.divide(attr_voc_sc_nhate,count_voc_sc_nhate)
    attr_voc_sc_sort_nhate = np.argsort(attr_voc_sc,axis=0)
    attr_voc_sc_sort_nhate = attr_voc_sc_sort_nhate[::-1] #reverse to sort in descending order
    tok_voc_sc_sort_nhate = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort_nhate[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort_nhate = [[tok_voc_sc_sort_nhate[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort_nhate[0]))]

    count_voc_sc_hate = np.where(count_voc_sc_hate == 0, 1, count_voc_sc_hate)
    attr_voc_sc = np.divide(attr_voc_sc_hate,count_voc_sc_hate)
    attr_voc_sc_sort_hate = np.argsort(attr_voc_sc,axis=0)
    attr_voc_sc_sort_hate = attr_voc_sc_sort_hate[::-1] #reverse to sort in descending order
    tok_voc_sc_sort_hate = [[dataset.tokenizer.convert_ids_to_tokens(idd) for idd in attr_voc_sc_sort_hate[:,ind].tolist()] for ind in range(6)]
    tok_col_voc_sort_hate = [[tok_voc_sc_sort_hate[ind][row] for ind in range(6)] for row in range(len(tok_voc_sc_sort_hate[0]))]

 
    with open(dataset_name+"_sorted_global_tokens.txt","w") as f:
      for s in tok_col_voc_sort:
          f.write(s[0]+"\t"+s[1]+"\t"+s[2]+"\t"+s[3]+"\t"+s[4]+"\t"+s[5])
          f.write("\n")

    with open(dataset_name+"_sorted_global_tokens_nhate.txt","w") as f:
      for s in tok_col_voc_sort_nhate:
          f.write(s[0]+"\t"+s[1]+"\t"+s[2]+"\t"+s[3]+"\t"+s[4]+"\t"+s[5])
          f.write("\n")

    with open(dataset_name+"_sorted_global_tokens_hate.txt","w") as f:
      for s in tok_col_voc_sort_hate:
          f.write(s[0]+"\t"+s[1]+"\t"+s[2]+"\t"+s[3]+"\t"+s[4]+"\t"+s[5])
          f.write("\n")

    with open(dataset_name+"_Imp_tok_IG.txt", "w") as f:
      for s in all_imp_tok_IG:
        f.write(str(s) +"\n")

    with open(dataset_name+"_Imp_tok_att.txt", "w") as f:
      for s in all_imp_tok_att:
        f.write(str(s) +"\n")

    with open(dataset_name+"_Imp_tok_attg.txt", "w") as f:
      for s in all_imp_tok_attg:
        f.write(str(s) +"\n")

    with open(dataset_name+"_Imp_tok_grad.txt", "w") as f:
      for s in all_imp_tok_grad:
        f.write(str(s) +"\n")

    with open(dataset_name+"_Imp_tok_omm.txt", "w") as f:
      for s in all_imp_tok_omm:
        f.write(str(s) +"\n")

    with open(dataset_name+"_Imp_tok_attmulg.txt", "w") as f:
      for s in all_imp_tok_attmulg:
        f.write(str(s) +"\n")
    
    with open(dataset_name+"_All_sent_wrdpieces.txt", "w") as f:
      for s in all_sent:
        s = " ".join(s)
        f.write(str(s) +"\n")

    with open(dataset_name+"_All_sent.txt", "w") as f:
      for s in all_sent_decode:
        f.write(str(s) +"\n")
     
    with open(dataset_name+"_pred_lab.txt","w") as f:
      for p in pred_lab:
        f.write(str(p) + "\n")   
 
    df = pd.DataFrame.from_dict(results_flip)
    

    df.to_csv(save_path + "_decision-flip-set.csv")
    
    
    df = df.drop(columns = "lengths")

    summary = df.mean(axis = 0)

    summary.to_csv(save_path + "_decision-flip-set-summary.csv", header = ["mean percentage"])

    all_sent = []
    all_sent_decode = []
    all_imp_tok_IG = []
    all_imp_tok_att = []
    all_imp_tok_attg = []
    all_imp_tok_omm = []
    all_imp_tok_attmulg = []
    all_imp_tok_grad = []
    pred_lab = []

    for sentences, lengths, labels in out_data:

        torch.cuda.empty_cache()
        model.zero_grad()
        model.eval()

        if args["encoder"] == "bert":

            sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)

        else:

            sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)

        # original trained model    
        model.train()

         

        yhat, weights_or = model(sentences, lengths, retain_gradient = True)

        yhat.max(-1)[0].sum().backward(retain_graph = True)

        pred_lab.extend(yhat.max(-1).indices.cpu().numpy())

        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)]

        integrated_grads = model.integrated_grads(sentences,
                                                g, lengths,
                                                original_pred = yhat.max(-1))


        weights_def_grad = model.weights.grad

        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]
        all_sent.extend(sent_tok)
        all_sent_decode.extend(dataset.tokenizer.batch_decode(sentences))

        with torch.no_grad():

            att_source_set = {}
            rand_set = {}
            att_grad_set = {}
            att_mul_grad_set = {}

            grad_set = {}
            ig_set = {}
            omission_set = {}

            g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            top_grad = torch.topk(g1, k = g1.size(1))[1].to(device)


            omission_scores = model.get_omission_scores(sentences, lengths, yhat)
            top_omission = torch.topk(omission_scores, k = weights_or.size(1))[1].to(device)

            integrated_grads.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            top_IG = torch.topk(integrated_grads, k = integrated_grads.size(1))[1].to(device)

            top_att = torch.topk(weights_or, k = weights_or.size(1))[1].to(device)
            top_randi = torch.randn(weights_or.shape)
            top_rand = torch.topk(top_randi, k = weights_or.size(1))[1].to(device)

            weight_mul_grad = weights_or * weights_def_grad

            weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))
            top_att_grad = torch.topk(weights_def_grad, k = weights_or.size(1))[1].to(device)


            weight_mul_grad.masked_fill_(model.masks[:,:max(lengths)].bool(),float("-inf"))
            top_att_mul_grad = torch.topk(weight_mul_grad, k = weights_or.size(1))[1].to(device)


            ind = 0 
            extr_tok_ig = []
            extr_tok_att = []
            extr_tok_attg = []
            extr_tok_grad = []
            extr_tok_grad = []
            extr_tok_omm = []
            extr_tok_attmulg = []


            sent_arr = [np.array(s) for s in sent_tok]
            extr_tok_ig = [list(sent_arr[ind][top_IG.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_att = [list(sent_arr[ind][top_att.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_attg = [list(sent_arr[ind][top_att_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_grad = [list(sent_arr[ind][top_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_omm = [list(sent_arr[ind][top_omission.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_attmulg = [list(sent_arr[ind][top_att_mul_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            extr_tok_ig = [list(sent_arr[ind][top_IG.cpu()[ind]]) for ind in range(len(sent_arr))]

            all_imp_tok_IG.extend(extr_tok_ig)
            all_imp_tok_att.extend(extr_tok_att)
            all_imp_tok_attg.extend(extr_tok_attg)
            all_imp_tok_grad.extend(extr_tok_grad)
            all_imp_tok_omm.extend(extr_tok_omm)
            all_imp_tok_attmulg.extend(extr_tok_attmulg)

    all_imp_tok_IG = [" ".join(s) for s in all_imp_tok_IG]
    all_imp_tok_att = [" ".join(s) for s in all_imp_tok_att]
    all_imp_tok_attg = [" ".join(s) for s in all_imp_tok_attg]
    all_imp_tok_grad = [" ".join(s) for s in all_imp_tok_grad]
    all_imp_tok_omm = [" ".join(s) for s in all_imp_tok_omm]
    all_imp_tok_attmulg = [" ".join(s) for s in all_imp_tok_attmulg]

    with open(out_dataset_name+"_Imp_tok_IG.txt", "w") as f:
       for s in all_imp_tok_IG:
           f.write(str(s) +"\n")

    with open(out_dataset_name+"_Imp_tok_att.txt", "w") as f:
       for s in all_imp_tok_att:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_Imp_tok_attg.txt", "w") as f:
       for s in all_imp_tok_attg:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_Imp_tok_grad.txt", "w") as f:
       for s in all_imp_tok_grad:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_Imp_tok_omm.txt", "w") as f:
       for s in all_imp_tok_omm:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_Imp_tok_attmulg.txt", "w") as f:
       for s in all_imp_tok_attmulg:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_All_sent_wrdpieces.txt", "w") as f:
       for s in all_sent:
          s = " ".join(s)
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_All_sent.txt", "w") as f:
       for s in all_sent_decode:
          f.write(str(s) +"\n")

    with open(out_dataset_name+"_pred_lab.txt","w") as f:
        for p in pred_lab:
          f.write(str(p) + "\n")
