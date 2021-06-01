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
import struct
import socket
import os
import datetime

def deepcod_send():
	client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# ADDR = ("127.0.0.1",8848)
	ADDR = ("130.126.136.154",8000)
	client.connect(ADDR)

	crs = [0.0039,0.0049,0.024,0.036,0.166]

	for cr in crs:
		size = int(224*224*3*cr)
		data = bytearray(os.urandom(size))
		data = struct.pack(">L", len(data))+data
		time_str = str(datetime.datetime.now())
		print(cr,size,len(data))
		client.send(str.encode(time_str)+data)
	client.close()

if __name__ == "__main__":
	np.random.seed(123)
	torch.manual_seed(2)

	# test 
	deepcod_send()

