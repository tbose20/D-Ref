import torch
import numpy as np
import pickle
from torch.utils.data import DataLoader
import pandas as pd
import json
import random
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def padder(ix_document, pad_len):
    
    seq_lens = []
    
    padded_all = []
    
    for doc in ix_document:
        
        if len(doc) == 0 :
           
            pass
               
        if (len(doc) < pad_len) & (len(doc) > 0):

            length = len(doc)
            
            diff = pad_len - len(doc)

            add_pad = [0]*diff

            padded = doc + add_pad

        elif len(doc) == pad_len:

            padded = doc
            
            length = len(doc)

        elif len(doc) > pad_len:

            padded = doc[:pad_len]
            
            length = pad_len
            
        padded_all.append(padded)
        seq_lens.append(length)
        
    return padded_all, seq_lens


class dataholder():
    
    def __init__(self, directory, dataset, B_SIZE  = 32):
        
        """
        Dataholder class (for non-bert instances)
        Accepts as input the data directory : directory
        The dataset : dataset
        and batch size: B_SIZE
        """
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        self.directory = directory
        self.dataset = dataset
        self.batch_size = B_SIZE
        self.hidden_dim = 64
        self.embedding_dim = 300
    
        all_data = pickle.load(open(directory + dataset + "/data.p", "rb"))
    
        self.w2ix = all_data.w2ix
        self.vocab_size = len(self.w2ix)    
        
        self.mask_list = []
        self.mask_tokens = ["<PAD>", "<SOS>", "<EOS>", "."]
        
        for item in self.mask_tokens:
            
            if item in self.w2ix:
                
                self.mask_list.append(self.w2ix[item])
        
        self.pretrained_embeds = all_data.pretrained_embeds
        
        
        # In[4]:
        
        
        x_train, y_train = zip(*all_data.train)
        x_dev, y_dev = zip(*all_data.dev)
        x_test, y_test = zip(*all_data.test)
        
        print("\nVocab size:", len(self.w2ix),
                "\nTraining size:", len(y_train),
                "\nDev size:", len(y_dev),
                "\nTest size:", len(y_test))
        
        # In[5]:
        
        self.output_size= len(np.unique(y_train))
        
        print("\nOutput dimension: ", self.output_size, "\n")
        
        
        self.sequence_length = all_data.sequence_length()
        
        if dataset == "mimicanemia":
        
        	self.sequence_length = 2200
        
        print("--Sequence length :", self.sequence_length, "\n")
        
        # In[10]:
        
        from modules.utils import padder
        
        x_train_pad, train_lengths = padder(x_train, pad_len = self.sequence_length)
        x_dev_pad, dev_lengths = padder(x_dev, pad_len = self.sequence_length)
        x_test_pad, test_lengths = padder(x_test, pad_len = self.sequence_length)
        
        
        # In[11]:
        
        x_train_pad = torch.LongTensor(x_train_pad)#.to(device)
        x_dev_pad = torch.LongTensor(x_dev_pad)#.to(device)
        x_test_pad = torch.LongTensor(x_test_pad)#.to(device)
        train_lengths = torch.LongTensor(train_lengths)#.to(device)
        dev_lengths =  torch.LongTensor(dev_lengths)#.to(device)
        test_lengths = torch.LongTensor(test_lengths)#.to(device)
        y_train = torch.LongTensor(y_train)#.to(device)
        y_dev = torch.LongTensor(y_dev)#.to(device)
        y_test = torch.LongTensor(y_test)#.to(device)
        
        
        # In[12]:
        
        
        training_prebatch = list(zip(x_train_pad, train_lengths, y_train))
        dev_prebatch = list(zip(x_dev_pad, dev_lengths, y_dev))
        testing_prebatch = list(zip(x_test_pad, test_lengths, y_test))
        
        
        training_prebatch = sorted(training_prebatch, key = lambda x : x[1], reverse = False)
        dev_prebatch = sorted(dev_prebatch, key = lambda x : x[1], reverse = False)
        testing_prebatch = sorted(testing_prebatch, key = lambda x : x[1], reverse = False)
        
        # In[13]:
        
        ### removing sos and eos only sentences
        
        train_prebatch = [x for x in training_prebatch if x[1] > 2]
        dev_prebatch = [x for x in dev_prebatch if x[1] > 2]
        test_prebatch = [x for x in testing_prebatch if x[1] > 2]
        
    
        self.training = DataLoader(train_prebatch, batch_size = self.batch_size, 
                                  shuffle = True, pin_memory = False)
            
        self.development = DataLoader(dev_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        
        
        self.testing = DataLoader(test_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        


def bertify(x, not_include = ["<SOS>", "<EOS>"]):
    
    bertification = []
    
    for word in x.split():
        
        if word == "<UNKN>":
            
            word = '[UNK]'
            
            bertification.append(word)
            
        elif word in not_include:
            
            pass
     
        else:
        
            bertification.append(word)
            
    return " ".join(bertification)
 
def bert_padder(x, max_len):
        
    if len(x) < max_len:
    
        x += [0]*(int(max_len) - len(x))
    
    elif len(x) > max_len:

        x = x[:max_len - 1]
        x += [102]

    return x

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

def mimic_halfer(x):

    stopwordlist = stopwords.words("english") + ["qqq", "<DE>", ":", ")", "(", ".", "/"]

    cut_no_stop = [word for word in x.split() if not word in stopwordlist]
    
    revised = cut_no_stop[20:276] + cut_no_stop[-256:]

    return " ".join(revised).lower()
        
from tqdm.auto import tqdm

class dataholder_bert():
    
    def __init__(self, directory, dataset, B_SIZE  = 8, bert_model = "bert-base_uncased"):        
        
        self.directory = directory
        self.dataset = dataset
        self.batch_size = B_SIZE
        self.hidden_dim = 768 // 2
        self.embedding_dim = None
        self.pretrained_embeds = None
        self.mask_list = [101, 102, 0]
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        
        all_data = pd.read_csv(directory + dataset + "/" + dataset + "_dataset.csv")

        all_data["text"] = all_data["text"].astype(str)
    
        labels = list(all_data["label"].values)
        ind_1 = [ind for ind in range(len(labels)) if labels[ind] == 1]
        ind_0 = [ind for ind in range(len(labels)) if labels[ind] == 0]

        #print("ind_1:",len(ind_1))
        #print("ind_0:",len(ind_0))


        #print("all_data:",all_data[:8])

        self.output_size = 2
        
        print("\nOutput dimension: ", self.output_size, "\n")
        
        from transformers import AutoTokenizer
       
        pretrained_weights = bert_model
       
        if dataset == "mimicanemia": bert_max_length = 512
        else: bert_max_length = 128 #To compare with Prior
        
        tokenizer = AutoTokenizer.from_pretrained(pretrained_weights, attention_window = bert_max_length,cache_dir='cache')
        
        self.tokenizer = tokenizer #TB

        self.vocab_size = tokenizer.vocab_size

        #print("tokenizer sentence: ",self.tokenizer.convert_tokens_to_ids("twilight feminist oppression"))

        # treat mimic for long text
        if dataset == "mimicanemia":
            # mimic requires reshuffling
            all_data = all_data.sample(frac = 1.0, random_state = 100)
            # borrowed idea from Fine-tune BERT for Text Classification 
            all_data["text"] = all_data["text"].apply(lambda x: mimic_halfer(x))
        
        # import pdb; pdb.set_trace();
        #remove sos and eos and replace unkn with bert symbols
        all_data["text"] = all_data["text"].apply(lambda x: bertify(x))
        # tokenize to maximum length the sequences and add the CLS token and ?SEP? at the enD??
        
        tqdm.pandas(desc="tokenizing")
        all_data["text"] = all_data["text"].progress_apply(lambda x: tokenizer.encode_plus(x, max_length = bert_max_length,truncation = True)["input_ids"])
        all_data["lengths"] = all_data["text"].apply(lambda x: len(x)) 
        
        self.sequence_length = int(all_data.lengths.quantile(q = 0.95))
            
        if self.sequence_length  < 50:
            
            self.sequence_length  = 50
        
        if dataset == "mimicanemia":
        
        	self.sequence_length  = 512
        
        
        print("--Sequence length :", self.sequence_length , "\n")
        
        if self.sequence_length  < 128:
            
            bert_max_length = self.sequence_length             
        
        all_data["text_encoded"] = all_data["text"].apply(lambda x: bert_padder(x, bert_max_length))

        train_prebatch = all_data[all_data.exp_split == "train"][["text_encoded", "lengths", "label"]].values.tolist()
        
        #indices = random.sample(range(len(train_prebatch)), 8000)
        #train_prebatch = [train_prebatch[ind] for ind in indices]
        #train_prebatch = train_prebatch[:-1]
        print("length train_prebatch: ",len(train_prebatch))

        dev_prebatch = all_data[all_data.exp_split == "dev"][["text_encoded", "lengths", "label"]].values.tolist()
        test_prebatch = all_data[all_data.exp_split == "test"][["text_encoded", "lengths", "label"]].values.tolist()
        
        # ### keep non zero sequences
      #  print("train_prebatch :",train_prebatch)
        
        train_prebatch = sorted(train_prebatch, key = lambda x : x[1], reverse = False)
        dev_prebatch = sorted(dev_prebatch, key = lambda x: x[1], reverse = False)
        test_prebatch = sorted(test_prebatch, key = lambda x: x[1], reverse = False)
        
        ### removing sos and eos only sentences
        
        train_prebatch = [x for x in train_prebatch if x[1] > 2]
        dev_prebatch = [x for x in dev_prebatch if x[1] > 2]
        test_prebatch = [x for x in test_prebatch if x[1] > 2]
        

        train_ids = [t[0] for t in train_prebatch]
        train_ids_hate = [t[0] for t in train_prebatch if t[2] == 1]
        train_ids_nhate = [t[0] for t in train_prebatch if t[2] == 0]


                            
        inv_doc_freq_dict = {}

        train_ids_uniq = set([ids for inner_list in train_ids for ids in inner_list])

        for id in list(train_ids_uniq):
            inv_doc_freq_dict[id] = 0
                                                                                            
            for ind_tr in train_ids:
                 if id in ind_tr:
                    inv_doc_freq_dict[id] += 1
                    
            inv_doc_freq_dict[id] = math.log(len(train_ids)/inv_doc_freq_dict[id])
        self.inv_doc_freq_dict = inv_doc_freq_dict

        inv_doc_freq_dict_hate = {}
        train_ids_uniq_hate = set([ids for inner_list in train_ids_hate for ids in inner_list])

        for id in list(train_ids_uniq_hate):
          inv_doc_freq_dict_hate[id] = 0
          
          for ind_tr in train_ids_hate:
              if id in ind_tr:
                  inv_doc_freq_dict_hate[id] += 1
          
          inv_doc_freq_dict_hate[id] = math.log(len(train_ids_hate)/inv_doc_freq_dict_hate[id])
        self.inv_doc_freq_dict_hate = inv_doc_freq_dict_hate



        inv_doc_freq_dict_nhate = {}
        train_ids_uniq_nhate = set([ids for inner_list in train_ids_nhate for ids in inner_list])

        for id in list(train_ids_uniq_nhate):
          inv_doc_freq_dict_nhate[id] = 0
          
          for ind_tr in train_ids_nhate:
              if id in ind_tr:
                  inv_doc_freq_dict_nhate[id] += 1
          inv_doc_freq_dict_nhate[id] = math.log(len(train_ids_nhate)/inv_doc_freq_dict_nhate[id])
        self.inv_doc_freq_dict_nhate = inv_doc_freq_dict_nhate




        self.training = DataLoader(train_prebatch, batch_size = self.batch_size, 
                      shuffle = False, pin_memory = False)
        
        self.development = DataLoader(dev_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)
        
        
        self.testing = DataLoader(test_prebatch, batch_size = self.batch_size, 
                               shuffle = False, pin_memory = False)

        
