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
	ADDR = ("130.126.136.154",8848)
	client.connect(ADDR)

	size = int(224*224*3*0.0039)
	data = bytearray(os.urandom(size))
	data = struct.pack(">L", len(data))+data
	time_str = str(datetime.datetime.now())
	client.send(str.encode(time_str)+data)
	client.close()

if __name__ == "__main__":
	np.random.seed(123)
	torch.manual_seed(2)

	# test 
	deepcod_send()

