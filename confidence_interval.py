from __future__ import division
import math, random
#from pycm import *
from tqdm import tqdm
import numpy
import sys
from sklearn import metrics
n_iterations = 10000
def sample(list_GT,list_dat):
  """Sample len(list) list elements with replacement"""

  resample_GT = []
  resample = []

  for _ in range(len(list_GT)//2):
    random_choice = random.choice(range(len(list_GT)))
    resample_GT.append(list_GT[random_choice])
    resample.append(list_dat[random_choice])

  return resample_GT, resample

def fscore(GT,data):
  """Compute F1 score"""
  return metrics.f1_score(GT,data,average='macro')

def confidence_interval(data_GT,data):

  print("len dat_GT:", len(data_GT))
  print("len data:", len(data))


  # create bootstrap distribution of F(B) - F(A)
  stats = []
  for i in range(n_iterations):
	# prepare train and test sets
    sample_GT, sample_data = sample(data_GT, data)
    stats.append(fscore(sample_GT, sample_data))
  
  # confidence intervals
  alpha = 0.95
  p = ((1.0-alpha)/2.0) * 100
  lower = max(0.0, numpy.percentile(stats, p))
  p = (alpha+((1.0-alpha)/2.0)) * 100
  upper = min(1.0, numpy.percentile(stats, p))
  print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha*100, lower*100, upper*100))
  print('GT F1:', fscore(data_GT, data))
