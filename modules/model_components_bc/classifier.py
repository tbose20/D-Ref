import torch
import torch.nn as nn
import math 
import json 
from torch.autograd import Variable
from sklearn.metrics import *
from sklearn import metrics
from tqdm import trange
from modules.run_experiments import *
import random
from torch import linalg as LA
from torch.autograd import grad
import numpy as np
import copy
import sys

sys.path.insert(1,'captum/')

from captum.attr import DeepLift
#import torch.Tensor as tensor

with open('modules/config.txt', 'r') as f:
    args = json.load(f)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Model(nn.Module):
    
    def __init__(self, encoder, attention,  mask_list,
                 hidden_dim, output_dim, tasc = None):
        super(Model, self).__init__()

        

        self.uniform = False
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim

        self.mask_list = mask_list

        self.encoder = encoder
        
        self.attention = attention
        
        #self.tasc_mech = tasc

        self.output = nn.Linear(hidden_dim*2, output_dim)

        stdv = 1. / math.sqrt(hidden_dim*2)
        self.output.weight.data.uniform_(-stdv, stdv)
        self.output.bias.data.fill_(0)
        

    def forward(self, embed, input, lengths=None, retain_gradient = False, ig = int(1)):
             
        self.hidden, last_hidden = self.encoder(embed, input, lengths=lengths, ig = ig)
        if retain_gradient:

            self.encoder.embed.retain_grad()

        masks = 0
     
        for item in self.mask_list:
        
            masks += (input == item)
        
        self.masks = masks
        self.lengths = lengths
        if lengths is None:
           self.weights = self.attention(self.hidden, masks)

        else:

           self.weights = self.attention(self.hidden, masks[:,:max(lengths)])
                
            
        
        last_layer = (self.weights.unsqueeze(-1)*self.hidden).sum(1)
 
        yhat = self.output(last_layer.squeeze(0))

        yhat = torch.softmax(yhat, dim = -1)
    
        
        return yhat.to(device)


    def get_omission_scores(self, input, lengths, predicted):

        input_pruned = input[:,:max(lengths)]

        omission_scores = []

        if len(predicted.shape) == 1:

            predicted = predicted.unsqueeze(0)

        predominant_class = predicted.max(-1)[1]
        self.eval()
        for _j in range(input_pruned.size(1)):
            torch.cuda.empty_cache()
            mask = torch.ones_like(input_pruned)
            mask[:,_j] = 0

            input_temp = input_pruned * mask

            ommited_pred = self.forward(input_temp, lengths)[0]

            if len(ommited_pred.shape) == 1:

                ommited_pred = ommited_pred.unsqueeze(0)

            ommited = ommited_pred[torch.arange(ommited_pred.size(0)), predominant_class]

            ommited[ommited != ommited] = 1

            scores = predicted.max(-1)[0] - ommited

            omission_scores.append(predicted.max(-1)[0] - ommited)

        omission_scores = torch.stack(omission_scores).transpose(0,1)

        return omission_scores

    def integrated_grads(self, original_input, original_grad, lengths, original_pred, steps = 20):

        grad_list = [original_grad]

        pred_list = []
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
            

            pred, _ = self.forward(original_input, lengths, ig = x)

            
                        
            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]
             
            pred_list.append(pred[rows,original_pred[1]].sum())


        pred_list = tuple(pred_list)
        g = grad(pred_list, self.encoder.embed,create_graph=True)
        g1 = original_grad.add(g[0])
        attributions = torch.divide(g1,len(pred_list)+1)
        self.zero_grad()


        em = self.encoder.embed

        ig = (attributions* em).sum(-1)[:,:max(lengths)]
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))


        return ig,em,attributions

    def integrated_grads_orig(self, original_input, original_grad, lengths, original_pred, steps = 20):

        grad_list = [original_grad]

        pred_list = []
        
        for x in torch.arange(start = 0.0, end = 1.0, step = (1.0-0.0)/steps):
           

            pred, _ = self.forward(original_input, lengths, retain_gradient = True, ig = x)

            
                        
            if len(pred.shape) == 1:

                pred = pred.unsqueeze(0)

            rows = torch.arange(pred.size(0))

            if x == 0.0:

                baseline = pred[rows, original_pred[1]]



            pred[rows, original_pred[1]].sum().backward()

            g = self.encoder.embed.grad

            grad_list.append(g)

        attributions = torch.stack(grad_list).mean(0)

        em = self.encoder.embed

        ig = (attributions* em).sum(-1)[:,:max(lengths)]
        
        self.approximation_error = torch.abs((attributions.sum() - (original_pred[0] - baseline).sum()) / pred.size(0))


        return ig,em,attributions



def train(model, training, development, loss_function, optimiser, run,epochs = 10, cutoff = True, save_folder  = None, cutoff_len = 2, data=None, out_data=None, vanilla = True):

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(100)
    np.random.seed(100)
    random.seed(100)
    torch.cuda.manual_seed_all(100)
  #  torch.use_deterministic_algorithms(True)

    results = []
    
    results_for_run = ""
    
    cut_off_point = 0

    filter_ids_FP = None

    filter_ids_FN = None

    
    for epoch in trange(epochs, desc = "run {}:".format(run+1), maxinterval = 0.1):

        itern = 0

        total_loss = 0
        total_L2_loss = 0
        total_cl_loss = 0

        tot_L2_loss = []
        cl_loss = []

        if args["encoder"] == "bert":
            model.encoder.bert.embeddings.requires_grad_(True)
        else:
            model.encoder.embedding.weight.requires_grad_(True)
        
        model.train()

        for sentences, lengths, labels in training:

           
               
            itern += 1
            
            model.zero_grad()
           
            if args["encoder"] == "bert":
            
                sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
            else:
                
                sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)


            if filter_ids_FP is not None:
                IND_drop = Variable(torch.zeros(len(sentences),len(sentences[0]))).to(device)
                for i in range(len(filter_ids_FP)):
                    ind_fill = (sentences == filter_ids_FP[i]).nonzero()
                    ones_fill = Variable(torch.ones(ind_fill.shape[0])).to(device)
                    #IND_drop.index_put_(tuple(ind_fill.t()), value)
                    IND_drop.index_put_(tuple(ind_fill.t()), ones_fill)

              #  for i in range(len(filter_ids_FP_TP)): 
              #      ind_fill = (sentences == filter_ids_FP_TP[i]).nonzero()
              #      ones_fill = Variable(torch.ones(ind_fill.shape[0])).to(device)
              #     #value = ones_fill*filt_list_FP_TP_sc[i]
              #      IND_drop.index_put_(tuple(ind_fill.t()), ones_fill)

                for i in range(len(filter_ids_FN)):
                    ind_fill = (sentences == filter_ids_FN[i]).nonzero()
                    ones_fill = Variable(torch.ones(ind_fill.shape[0])).to(device)
                    #IND_drop.index_put_(tuple(ind_fill.t()), value)
                    IND_drop.index_put_(tuple(ind_fill.t()), ones_fill)


            
            inpt_seq = sentences.long() 

            model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)

            

            yhat_i = model(model.encoder.embed, sentences, lengths)
           
            em = model.encoder.embed


            if filter_ids_FP is not None:

                dl = DeepLift(model)
                masks = model.masks
                baseline = torch.zeros(model.encoder.embed.size())
                baseline = baseline.to(device)
                dl_attr = dl.attribute(model.encoder.embed, additional_forward_args = sentences, baselines = baseline, target= yhat_i.max(-1).indices)
                dl_attr = dl_attr.sum(-1)[:,:max(lengths)]


                dl_attr.masked_fill_(masks[:,:max(lengths)].bool(), 0)  #As masks should not contribute
                         
                dl_attr[dl_attr != dl_attr] = 0.0
                
             
            if filter_ids_FP is not None:

                     IND_drop = IND_drop[:,:max(lengths)]



                     target_attr = dl_attr * (1-IND_drop)

                     diff = dl_attr - target_attr
                     L2_loss = LA.norm(diff, dim = 1)

                     L2_loss = torch.mean(L2_loss)

                     L2_loss = args["alpha"]*L2_loss

                     total_L2_loss +=  args["alpha"]  * L2_loss.item()
 
                     L2_loss.backward()
                     


            inpt_seq = sentences.long() 

            model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)
 
            yhat =  model(model.encoder.embed, sentences, lengths)
            em = model.encoder.embed

            if len(yhat.shape) == 1:
                
                yhat = yhat.unsqueeze(0)

           
            loss = loss_function(yhat, labels)
            loss1 = loss  
            total_cl_loss += loss1.item()
            cl_loss.append(loss1.item())

            total_loss += float(loss.item())
 
            loss.backward()   

            _, ind = torch.max(yhat, dim = 1)

            optimiser.step()
        
       
        dev_results, dev_loss,_,_ = test(model, loss_function, development)    

        results.append([epoch, dev_results["macro avg"]["f1-score"], dev_loss, dev_results])
        
        results_for_run += "epoch - {} | train loss - {}| classification loss - {} | dev f1 - {} | dev loss - {} \n".format(epoch + 1,
                                    round(total_loss * training.batch_size / len(training),2),
                                    round(total_cl_loss * training.batch_size / len(training),2),
                                    round(dev_results["macro avg"]["f1-score"], 3),
                                    round(dev_loss, 2))

        print(results_for_run)
        if filter_ids_FP is not None:
            print("total_L2_loss: ",total_L2_loss)
            print("L2 loss: ",round(total_L2_loss * training.batch_size / len(training),8))
        
        if save_folder is not None:
            
            if epoch == 0:
                print("Epoch 0 pass")
            elif epoch ==1:
             
                torch.save(model.state_dict(), save_folder)
                best_model = model

                saved_model_results  = dev_results
                saved_model_results["training_loss"] = total_loss * training.batch_size / len(training)
                saved_model_results["epoch"] = epoch+1
                saved_model_results["dev_loss"] = dev_loss
                
            else:
                

                if saved_model_results["macro avg"]["f1-score"] < dev_results["macro avg"]["f1-score"]:
   
                    torch.save(model.state_dict(), save_folder)
                    best_model = model
                    saved_model_results  = dev_results
                    saved_model_results["training_loss"] = total_loss * training.batch_size / len(training)
                    saved_model_results["epoch"] = epoch+1
                    saved_model_results["dev_loss"] = dev_loss


        ## cutoff
        if cutoff == True:
         
            if len(results) > cutoff_len:
         
                diff = results[-1][2] - results[-2][2]

                if diff > 0:
                    
                    cut_off_point += 1
                    
                else:
                    
                    cut_off_point = 0
                    
        if cut_off_point == cutoff_len:
            
            break
      
        print("epoch: ",epoch)
        if epoch == 0:
            evaluation = evaluate(classifier = model,
                      loss_function = loss_function,
                      data = [args["dataset"], data],
                      out_data = [args["out_dataset"],out_data],
                      save_path = args["experiments_path"],
                      epoch=epoch
                      )
        elif not vanilla: 
            evaluation = evaluate(classifier = model,  
                      loss_function = loss_function,
                      data = [args["dataset"], data],
                      out_data = [args["out_dataset"],out_data],
                      save_path = args["experiments_path"],
                      epoch=epoch,
                      FP = filt_list_FP,
                      FN = filt_list_FN
                      )
        if not vanilla:

           filt_list_FP,_,filt_list_FN,_  = evaluation.decision_flip_set()
           filter_ids_FP = data.tokenizer.convert_tokens_to_ids(filt_list_FP)  

           filter_ids_FP = Variable(torch.Tensor(filter_ids_FP)).to(device)

           filter_ids_FN = data.tokenizer.convert_tokens_to_ids(filt_list_FN)  

           filter_ids_FN = Variable(torch.Tensor(filter_ids_FN)).to(device)

    return best_model, saved_model_results, results_for_run

def test(model, loss_function, data):
    
    predicted = [] 
    
    actual = []
    
    total_loss = 0
    
    model.eval()
    
    with torch.no_grad():

        for sentences, lengths, labels in data:
    
            if args["encoder"] == "bert":
            
                sentences, lengths, labels = torch.stack(sentences).transpose(0,1).to(device), lengths.to(device), labels.to(device)
                
            else:
                
                sentences, lengths, labels = Variable(sentences).to(device),Variable(lengths).to(device), Variable(labels).to(device)
            
            inpt_seq = sentences.long() 

            model.encoder.embed = model.encoder.bert.embeddings.word_embeddings(inpt_seq)


            yhat =  model(model.encoder.embed, sentences, lengths)
            
            if len(yhat.shape) == 1:
    
                yhat = yhat.unsqueeze(0)
    
            loss = loss_function(yhat, labels)
        
            total_loss += loss.item()
            
            _, ind = torch.max(yhat, dim = 1)
    
            predicted.extend(ind.cpu().numpy())
    
            actual.extend(labels.cpu().numpy())
       
        


        results = classification_report(actual, predicted, output_dict = True)
        microF1 = metrics.f1_score(actual,predicted,average='micro')
        macroF1 = metrics.f1_score(actual,predicted,average='macro')
        weightedF1 = metrics.f1_score(actual,predicted,average='weighted')
        accuracy = metrics.accuracy_score(actual,predicted)
        print(results)
        print("microF1: ",microF1," macroF1: ",macroF1," weightedF1: ",weightedF1," accuracy: ",accuracy)
        print(confusion_matrix(actual,predicted))
 
    
    return results, (total_loss * data.batch_size / len(data)), actual, predicted 

