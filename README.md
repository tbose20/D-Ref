## D-Ref

Repository for the paper *"Dynamically Refined Regularization for Improving Cross-corpora Hate Speech Detection", Tulika Bose, Nikolaos Aletras, Irina Illina and Dominique Fohr, at Findings of ACL 2022*. Paper available at this [link](https://aclanthology.org/2022.findings-acl.32/)

## Prerequisites

Install necessary packages by using   

```
conda env create -f environment.yml
conda activate D-Ref
pip install transformers

```

## Getting the Data
Please note that we are not allowed to distribute the datasets. As such, please follow these instructions to retreive the presented datasets:

   Dynamic: The dataset used for the experiments in the paper is an older version of the one present [here](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset). Since the older version is no longer available, the latest version of the dataset can be downloaded from the link above.
  
   HatEval:  Please fill [this](http://hatespeech.di.unito.it/hateval.html) form to submit a request to the authors.
   
   Waseem: The dataset is available as TweetIDs [here](https://github.com/zeeraktalat/hatespeech) as NAACL_SRW_2016.csv. Note that the dataset used for the experiments may not exactly match the dataset obtained after crawling the TweetIDs as tweets keep getting removed over time.


## Training and Evaluating the models
1. The dataset should be prepared in the same format as used by this example dataset [tasks/sst/sst_dataset.csv](https://github.com/tbose20/D-Ref/blob/master/tasks/sst/sst_dataset.csv). The name of the data file should follow the naming convention of \<name of dataset sub-directory\>\_dataset.csv, e.g. sst/sst_dataset.csv.

2. Run the script run_tuning.sh for obtaining the optimal values of the hyper-parameters 'alpha' and 'INP_per' that correspond to the lambda and k, respectively. Please use the validation set performances obtained by running the script for each pair of 'alpha' and 'INP_per' to find the optimal values.

```
sh run_tuning.sh > tuning.txt 
```
3. You can train and save the models with [train_eval_bc.py]() script with the following options:


* dataset : *{HatEval, Dynamic, Waseem}* The train sub-part in the dataset file should be the training set from the source corpus, whereas the validation sub-part should be the validation set from the target corpus (which is passed to the 'out-dataset' argument too). For obtaining in-corpus performance, the test subpart of this file can be kept as the test set of the source corpus. 
* out_dataset : *{HatEval, Dynamic, Waseem}* The train, validation and test subparts should belong to the target corpus.
* encoder : *bert* 
* data_dir : *directory where the datasets are present* 
* model_dir : *directory for saved models*
* vanilla: flag to obtain the baseline results for BERT Van-FT. While running D-Ref, do not use this flag.

Example script

``` 
python -u train_eval_bc.py -dataset HatEval -out_dataset  Dynamic -encoder bert -data_dir tasks/ -model_dir models/ -alpha $alpha -perc_inp $INP_per

```

This code is adapted from the repository [here](https://github.com/GChrysostomou/tasc/tree/ed1a421b3cff68e8023d605e384573b07b6c81d6)
