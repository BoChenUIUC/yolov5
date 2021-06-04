import cv2
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from app import Simulator
from compression.transformer import Transformer

def speed_test(EXP_NAME):
	np.random.seed(123)
	torch.manual_seed(2)

	sim = Simulator(train=False)
	TF = Transformer(name=EXP_NAME)
	datarange = [66,70]
	eval_file = open(EXP_NAME+'_spdtest.log', "w", 1)

	if EXP_NAME == 'JPEG':
		rate_ranges = [7,11,15,21,47,100]
	elif EXP_NAME == 'WebP':
		rate_ranges = [0,5,37,100]
	elif EXP_NAME == 'Scale':
		rate_ranges = [0.1*x for x in range(1,11)]
	for r in rate_ranges:
		acc,cr = sim.get_one_point(datarange, TF=TF, C_param=r)
		print(EXP_NAME,r,TF.get_compression_time())
	m,s = TF.get_compression_time()
	eval_file.write(f"{m:.5f} {s:.5f}\n")

def test():
	from app import deepcod_validate,deepcod_avgsize
	# deepcod_validate()
	deepcod_avgsize()

if __name__ == "__main__":
	np.random.seed(123)
	torch.manual_seed(2)

	# test 
	test()

	# for name in ['JPEG','Scale','WebP']:
	# 	speed_test(name)
