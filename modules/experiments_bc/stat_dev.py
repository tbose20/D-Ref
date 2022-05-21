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
import tqdm

"""MODEL OUT DISTRIBUTIONS"""

with open('modules/config.txt', 'r') as f:
    args = json.load(f)


def stat_filter(data,dataset,glob_list):
    
  #  all_data = pd.read_csv(data_dir + dataset + "/" + dataset + "_dataset.csv")
  #  all_data["text"] = all_data["text"].apply(lambda x: bertify(x))
  #      # tokenize to maximum length the sequences and add the CLS token and ?SEP? at the enD??
  #  
  #  tqdm.pandas(desc="tokenizing dev set")
  #  all_data["text"] = all_data["text"].progress_apply(lambda x: dataset.tokenizer.encode_plus(x, max_length = bert_max_length,truncation = True)["input_ids"])
  #  all_data["lengths"] = all_data["text"].apply(lambda x: len(x))
  #  sequence_length = int(all_data.lengths.quantile(q = 0.95))

  #  if self.sequence_length  < 50:

  #     self.sequence_length  = 50

  #  if self.sequence_length  < 512:

  #     bert_max_length = sequence_length

  #  all_data["text_encoded"] = all_data["text"].apply(lambda x: bert_padder(x, bert_max_length))

  #  dev_inst = all_data[all_data.exp_split == "dev"][["text_encoded", "lengths", "label"]].values.tolist()
    #filt_tokens = []
    filt_tokens_FP = []
    filt_tokens_TP = []

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
        
    
        yhat, weights_or = model(sentences, lengths, retain_gradient = True)
        
        yhat.max(-1)[0].sum().backward(retain_graph = True)
        
       # print("yhat.max(-1):",yhat.max(-1))
       # print("yhat.max(-1).indices:",yhat.max(-1).indices)

        pred_lab = yhat.max(-1).indices.cpu().numpy()


        #g = model.encoder.embed.grad

        #em = model.encoder.embed

        #g1 = (g* em).sum(-1)[:,:max(lengths)]

        #integrated_grads = model.integrated_grads(sentences, 
        #                                        g, lengths, 
        #                                        original_pred = yhat.max(-1))
         

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
            
            #g1.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))

            #top_grad = torch.topk(g1, k = g1.size(1))[1].to(device)
     
            top_att = torch.topk(weights_or, k = weights_or.size(1))[1].to(device)
            #top_randi = torch.randn(weights_or.shape)
            #top_rand = torch.topk(top_randi, k = weights_or.size(1))[1].to(device)

            weight_mul_grad = weights_or * weights_def_grad

            #weights_def_grad.masked_fill_(model.masks[:,:max(lengths)].bool(), float("-inf"))

            #top_att_grad = torch.topk(weights_def_grad, k = weights_or.size(1))[1].to(device)
            #extr_tok_attg = [list(sent_arr[ind][top_att_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            
            sent_arr = [np.array(s) for s in sent_tok]

            top_att_mul_grad = torch.topk(weight_mul_grad, k = weights_or.size(1))[1].to(device)
            extr_tok_attg = [list(sent_arr[ind][top_att_grad.cpu()[ind]]) for ind in range(len(sent_arr))]
            
            top_inst_tok = [ext[:5] for ext in extr_tok_attg]
            
            lab = labels.cpu().numpy()
            for inst in len(pred_lab):
                if pred_lab[inst] == 1 and lab[inst] == 0:
                    filt_tokens_FP.extend(list(set(top_inst_tok[inst]) & set(glob_list)))
            for inst in len(pred_lab):
                if pred_lab[inst] == 1 and lab[inst] == 1:
                    filt_tokens_TP.extend(list(set(top_inst_tok[inst]) & set(glob_list)))
            filt_tok_FP_only = [tok for tok in filt_tokens_FP if tok not in filt_tokens_TP]
            filt_tok_TP_FP = [tok for tok in filt_tokens_FP if tok in filt_tokens_TP]

    return filt_tok_FP_only, filt_tok_TP_FP, filt_tokens_TP

#def regularizer():
    

def perc_remove(data, out_data, model, save_path,dataset, dataset_name, out_dataset_name):
    
    #for sentences, lengths, labels in data:
          


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
        
    
        yhat, weights_or = model(sentences, lengths, retain_gradient = True)
        
        yhat.max(-1)[0].sum().backward(retain_graph = True)
        
       # print("yhat.max(-1):",yhat.max(-1))
       # print("yhat.max(-1).indices:",yhat.max(-1).indices)

        pred_lab.extend(yhat.max(-1).indices.cpu().numpy())


        ################## Uncomment from here ######################
        g = model.encoder.embed.grad

        em = model.encoder.embed

        g1 = (g* em).sum(-1)[:,:max(lengths)]

        integrated_grads = model.integrated_grads(sentences, 
                                                g, lengths, 
                                                original_pred = yhat.max(-1))
         

        weights_def_grad = model.weights.grad
        
        #print("data:",dataset)
        #print("sentences: ",sentences)
        #sent_tok = []
        #for sent in sentences:
        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]
        #sent_tok = dataset.tokenizer.batch_decode(sentences) #TB
        all_sent.extend(sent_tok)
        all_sent_decode.extend(dataset.tokenizer.batch_decode(sentences))
        #print("sent_tok:",sent_tok)

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

             
            ind = 0 #TB
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


           # for s in sent_tok:  #TB
           #    #print("s:",s)
           #    s = np.array(s.split())  #TB
           #    sort_s_ig = list(s[top_IG[ind]]) #TB
           #    sort_s_att = list(s[top_att[ind]]) #TB
           #    sort_s_attg = list(s[top_att_grad[ind]]) #TB
           #    sort_s_grad = list(s[top_grad[ind]]) #TB
           #    sort_s_omm = list(s[top_omission[ind]]) #TB
           #    sort_s_attmulg = list(s[top_att_mul_grad[ind]]) #TB
           #    extr_tok_ig.append(sort_s_ig) #TB
           #    extr_tok_att.append(sort_s_att) #TB
           #    extr_tok_attg.append(sort_s_attg) #TB
           #    extr_tok_grad.append(sort_s_grad) #TB
           #    extr_tok_omm.append(sort_s_omm) #TB
           #    extr_tok_attmulg.append(sort_s_attmulg) #TB
           #    ind+=1  #TB

            all_imp_tok_IG.extend(extr_tok_ig)
            all_imp_tok_att.extend(extr_tok_att)
            all_imp_tok_attg.extend(extr_tok_attg)
            all_imp_tok_grad.extend(extr_tok_grad)
            all_imp_tok_omm.extend(extr_tok_omm)
            all_imp_tok_attmulg.extend(extr_tok_attmulg)

            #print("sent_tok:",sent_tok)
            #print("top_IG:",top_IG)
            #print("all_imp_tok:",all_imp_tok)

            
            temp = 0
            
            model.eval()

            maximum = max(lengths)
            increments =  torch.round(maximum.float() * 0.02).int()
            
            ## to speed up experiments if you want
            if args["speed_up"]:
            
                increments = max(1,increments)

            else:

                increments = 1
            ################### Uncomment upto here #########################


         #   lengths_ref = lengths.clone()
         #   
         #   maximum = max(lengths)
         #   
         #   lengths_ref = lengths.clone()
         #   
         #   rows = torch.arange(sentences.size(0)).long().to(device)

         #   original_sentences = sentences.clone().detach()

         #   for _j_ in range(0,maximum, increments):
         #       
         #       """Attention at source"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #      
         #       mask = mask.scatter_(1,  top_att[rows, _j_+1:], 1)
         #   
         #       yhat_max_source, _ = model(sentences.float() * mask.float(), lengths)
         #       
         #       check_indexes_max_s = (yhat.max(-1)[1] != yhat_max_source.max(-1)[1]).nonzero()
         #           
         #       if check_indexes_max_s.nelement() != 0:
         #           for atin in check_indexes_max_s:
         #  
         #               
         #               if atin.item() not in att_source_set.keys():
         #                   
         #                   temp += 1
         #                   att_source_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
         #                   
         #               else:
         #                   
         #                   pass
         #               
         #       """Attention gradient at source"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #      
         #       mask = mask.scatter_(1,  top_att_grad[rows, _j_+1:], 1)
         #   
         #       yhat_grad_att_source, _ = model(sentences.float() * mask.float(), lengths)
         #       
         #       check_indexes_grad_att_s = (yhat.max(-1)[1] != yhat_grad_att_source.max(-1)[1]).nonzero()
         #           
         #       if check_indexes_grad_att_s.nelement() != 0:
         #           for atin in check_indexes_grad_att_s:
         #  
         #               
         #               if atin.item() not in att_grad_set.keys():
         #                   
         #                   temp += 1
         #                   att_grad_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
         #                   
         #               else:
         #                   
         #                   pass
         #               
         #       """Attention * gradient at source"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #      
         #       mask = mask.scatter_(1,  top_att_mul_grad[rows, _j_+1:], 1)
         #       
         #       yhat_grad_mul_att_source, _ = model(sentences.float() * mask.float(), lengths)
         #       
         #       check_indexes_grad_mul_att_s = (yhat.max(-1)[1] != yhat_grad_mul_att_source.max(-1)[1]).nonzero()
         #           
         #       if check_indexes_grad_mul_att_s.nelement() != 0:
         #           for atin in check_indexes_grad_mul_att_s:
         #  
         #               if atin.item() not in att_mul_grad_set.keys():
         #                   
         #                   temp += 1
         #                   att_mul_grad_set[atin.item()] = (_j_ + 1) / lengths_ref[atin].item() 
         #                   
         #               else:
         #                   
         #                   pass

         #     
         #       """Gradient"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #   
         #       mask = mask.scatter_(1,  top_grad[rows, _j_+1:], 1)
         #   
         #       yhat_grad,_ = model(sentences.float() * mask.float(), lengths)
    
         #       check_indexes_grad = (yhat.max(-1)[1] != yhat_grad.max(-1)[1]).nonzero()
         #       
         #       if check_indexes_grad.nelement() != 0:
         #           
         #           for items in check_indexes_grad:
         #                           
         #               if items.item() not in grad_set.keys():
         #           
         #                   temp += 1
         #                   grad_set[items.item()] = (_j_ + 1) / lengths_ref[items].item()
         #                   
         #               else:
         #                   
         #                   pass


         #       """INTEGRATED Gradient"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #   
         #       mask = mask.scatter_(1,  top_IG[rows, _j_+1:], 1)
         #   
         #       yhat_IG,_ = model(sentences.float() * mask.float(), lengths)
    
         #       check_indexes_IG = (yhat.max(-1)[1] != yhat_IG.max(-1)[1]).nonzero()
         #       
         #       if check_indexes_IG.nelement() != 0:
         #           
         #           for items_IG in check_indexes_IG:
         #                           
         #               if items_IG.item() not in ig_set.keys():
         #           
         #                   temp += 1
         #                   ig_set[items_IG.item()] = (_j_ + 1) / lengths_ref[items_IG].item()
         #                   
         #               else:
         #                   
         #                   pass
    


         #       """Ommision"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #   
         #       mask = mask.scatter_(1,  top_omission[rows, _j_+1:], 1)
         #       
         #       yhat_omission, _ = model(sentences.float() * mask.float(), lengths)
         #       
         #       check_indexes_omission = (yhat.max(-1)[1] != yhat_omission.max(-1)[1]).nonzero()
         #           
         #       if check_indexes_omission.nelement() != 0:
         #           for omi in check_indexes_omission:
        
         #               if omi.item() not in omission_set.keys():
         #                   
         #                   temp += 1
         #                   omission_set[omi.item()] = (_j_ + 1) / lengths_ref[omi].item() 
         #                   
         #               else:
         #                   
         #                   pass
         #       
    
         #       """Random"""
         #       
         #       mask = torch.zeros(sentences.shape).to(device)
         #      
         #       mask = mask.scatter_(1,  top_rand[rows, _j_+1:], 1)
         #       
         #       
         #       yhat_rand, _ = model(sentences.float() * mask.float(), lengths)
         #       
         #       check_indexes_rand  = (yhat.max(-1)[1] != yhat_rand.max(-1)[1]).nonzero()
         #       
         #       if check_indexes_rand.nelement() != 0:
         #           
         #           for rna in check_indexes_rand:
         #                                   
         #               if rna.item() not in rand_set.keys():
         #            
         #                   rand_set[rna.item()] = (_j_ + 1) / lengths_ref[rna].item()
         #                   
         #               else:
         #                   
         #                   pass
         #               

         #   
         #   for _i_ in range(0, sentences.size(0)):
         #       
         #       if _i_ not in rand_set.keys():
         #           
         #           rand_set[_i_] = 1
       
         #       if _i_ not in att_source_set.keys():
         #           
         #           att_source_set[_i_] = 1
         #   
         #       if _i_ not in att_grad_set.keys():
         #           
         #           att_grad_set[_i_] = 1
         #           
         #       if _i_ not in att_mul_grad_set.keys():
         #           
         #           att_mul_grad_set[_i_] = 1
         #      
         #       if _i_ not in omission_set.keys():
         #           
         #           omission_set[_i_] = 1

         #           
         #       if _i_ not in grad_set.keys():
         #           
         #           grad_set[_i_] = 1

         #       if _i_ not in ig_set.keys():
         #           
         #           ig_set[_i_] = 1

         #           

         #   
         #   att_mul_grad_set = {k:(1 if v > 1 else v) for k,v in att_mul_grad_set.items()}
         #   att_grad_set = {k:(1 if v > 1 else v) for k,v in att_grad_set.items()}
         #   rand_set = {k:(1 if v > 1 else v) for k,v in rand_set.items()}
         #   att_source_set = {k:(1 if v > 1 else v) for k,v in att_source_set.items()}
         #   

         #   att_mul_grad_set = OrderedDict(sorted(att_mul_grad_set.items()))
         #   att_grad_set = OrderedDict(sorted(att_grad_set.items()))
         #   rand_set = OrderedDict(sorted(rand_set.items()))
         #   att_source_set = OrderedDict(sorted(att_source_set.items()))

         #   ig_set = {k:(1 if v > 1 else v) for k,v in ig_set.items()}
         #   grad_set = {k:(1 if v > 1 else v) for k,v in grad_set.items()}
         #   omission_set = {k:(1 if v > 1 else v) for k,v in omission_set.items()}
         #   
         #   ig_set = OrderedDict(sorted(ig_set.items()))
         #   grad_set = OrderedDict(sorted(grad_set.items()))
         #   omission_set = OrderedDict(sorted(omission_set.items()))

         #   if len(yhat.shape) == 1:
         #       
         #       pass
         #       
         #   else:
         #       
         #       
         #       results_flip["random"].extend(rand_set.values())
         #       results_flip["max_source"].extend(att_source_set.values())
         #       results_flip["att_grad"].extend(att_grad_set.values())
         #       results_flip["lengths"].extend(lengths.cpu().data.numpy())
         #       results_flip["att*grad"].extend(att_mul_grad_set.values())

         #       results_flip["grad"].extend(grad_set.values())
         #       results_flip["omission"].extend(omission_set.values())
         #       results_flip["IG"].extend(ig_set.values())
    
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
    print("Arg sort..........:",attr_voc_sc_sort_hate)
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

        #print("data:",dataset)
        #print("sentences: ",sentences)
        #sent_tok = []
        #for sent in sentences:
        sent_tok = [dataset.tokenizer.convert_ids_to_tokens(sent) for sent in sentences]
        #sent_tok = dataset.tokenizer.batch_decode(sentences) #TB
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


            ind = 0 #TB
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
