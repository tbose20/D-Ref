import modules.experiments_bc.decision_single as eoo
import modules.experiments_bc.decision_set as perc
import modules.experiments_bc.set_tp as dmp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class evaluate:
    
    def __init__(self, classifier, loss_function, data, out_data, save_path, bayesian = None, epoch=None, FP = None, FN = None):
        
        self.classifier = classifier
        self.loss_function = loss_function
        self.dataset_name, dataset = data
        self.out_dataset_name, out_dataset = out_data
        self.dataset = dataset 
        self.out_dataset = out_dataset
        self.training = dataset.training 
        self.development = dataset.development
        self.out_development = out_dataset.development
        self.out_testing = out_dataset.testing  
        self.save_path = save_path
        self.bayesian = bayesian
        self.epoch = epoch
        
        self.encode_sel = classifier.encoder.encode_sel
        self.mechanism_name = classifier.attention.__class__.__name__
       
        self.save_dir = self.save_path + \
        self.encode_sel + "_" + self.mechanism_name
        self.FP = FP
        self.FN = FN

    def decision_flip_single(self):
        
        """
        Removing the most informative token
        """
        
        eoo.effect_on_output(self.testing, 
                   self.classifier, 
                   save_path = self.save_dir)
        
        
    def decision_flip_set(self):
        
        """
        Recording the fraction of tokens required to cause a decision flip
        """
        glob_all_list,glob_nhate_list,glob_hate_list,_ = perc.extract_ftr_attr(self.training, self.out_testing, self.classifier, self.epoch, dataset=self.dataset, dataset_name=self.dataset_name, out_dataset_name=self.out_dataset_name)
        glob_hate_list = [glob[2] for glob in glob_hate_list]  #Kept space for other ftr attr in glob_hate_list, even when using one
        glob_all_list = [glob[2] for glob in glob_all_list]
        glob_nhate_list = [glob[2] for glob in glob_nhate_list]
  

        glob_nhate_list2 = glob_nhate_list

        glob_list = glob_hate_list[:500]
        glob_nhate_list = glob_nhate_list[:500]

        filt_list_FP_only, filt_list_TP, sent_FP,filt_list_FN_only, filt_list_TN, sent_FN = perc.stat_filter(self.development, self.classifier, dataset=self.dataset,glob_list=glob_list, glob_nhate_list = glob_nhate_list)

        filt_list_FP = set(filt_list_FP_only) #Remove duplicate
        filt_list_TP = set(filt_list_TP)


        filt_list_FP = list(filt_list_FP)
        filt_list_TP = list(filt_list_TP)


        filt_list_FN = set(filt_list_FN_only) #Remove duplicate
        filt_list_TN = set(filt_list_TN)


        filt_list_FN = list(filt_list_FN)
        filt_list_TN = list(filt_list_TN)

        # Sort the filtered list based on global rank
        ind_glob1 = [glob_list.index(tok) for tok in filt_list_FP]

        ind_glob1 = sorted(ind_glob1) #In ascending order as smaller the rank, higher the attr
        filt_list_FP = [glob_list[ind] for ind in ind_glob1]

        ind_glob1 = [glob_nhate_list.index(tok) for tok in filt_list_FN]

        ind_glob1 = sorted(ind_glob1) #In ascending order as smaller the rank, higher the attr
        filt_list_FN = [glob_nhate_list[ind] for ind in ind_glob1]

        return filt_list_FP, filt_list_TP, filt_list_FN, filt_list_TN
    

    def correct_classified_set(self, data_size, largest = True):
        """
        Conducts a decision flip experiment
        on correctly classified instances
        """
        dmp.degrading_model_perf(data = self.testing, 
            model = self.classifier, 
            save_path  = self.save_dir,
            data_size = data_size,
            largest = largest)


    def topbotk_s(self, w2ix, k = 10):
        """
        returns scores for explanations of topkwords
        """
        
        assert self.classifier.attention == True  
        
        explanations = dict(enumerate((self.classifier.explanations.unsqueeze(-1) * self.classifier.encoder.embedding.weight).sum(-1) ))
 
        ix2w = {v:k for k,v in w2ix.items()}
        
        word_scores = tuple({ix2w[k]:v.item() for k,v in explanations.items()}.items())
        
        top_words = list(sorted(word_scores, key = lambda x: x[1], reverse = True))[:k]
        bottom_words = list(sorted(word_scores, key = lambda x: x[1], reverse = False))[:k]
        bottom_words = list(sorted(bottom_words, key = lambda x: x[1], reverse = True))
        
        total = top_words + bottom_words
        
        plt.clf()
        df = pd.DataFrame(total)
        df.to_csv(self.save_dir + "_topbottomk.csv")
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1, 1, 1)
        
        
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        sns.barplot(data= df, x = 1, y = 0)
        plt.xlabel("scores (u)")
        plt.ylabel("")
        
        plt.savefig(self.save_dir + "_topbottomk_s.png")
        
    def topbotk_u(self, w2ix, k = 10):
        """
        returns scores for explanations of topkwords
        """
        
        assert self.classifier.tasc_attention == True
        
        explanations = dict(enumerate((self.classifier.explanations)))
 
        ix2w = {v:k for k,v in w2ix.items()}
        
        word_scores = tuple({ix2w[k]:v.item() for k,v in explanations.items()}.items())
        
        top_words = list(sorted(word_scores, key = lambda x: x[1], reverse = True))[:k]
        bottom_words = list(sorted(word_scores, key = lambda x: x[1], reverse = False))[:k]
        bottom_words = list(sorted(bottom_words, key = lambda x: x[1], reverse = True))
        
        total = top_words + bottom_words
        
        plt.clf()
        df = pd.DataFrame(total)
        df.to_csv(self.save_dir + "_topbottomk.csv")
        fig = plt.figure(figsize = (10,10))
        ax = fig.add_subplot(1, 1, 1)
        
        
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        sns.barplot(data= df, x = 1, y = 0)
        plt.xlabel("scores (u)")
        plt.ylabel("")
        
        plt.savefig(self.save_dir + "_topbottomk_u.png")
