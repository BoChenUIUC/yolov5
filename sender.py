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
	payload_size = struct.calcsize(">L")
	# ADDR = ("127.0.0.1",8848)
	ADDR = ("130.126.136.154",8000)
	client.connect(ADDR)

	crs = [0.173/32,0.312,0.236/32,0.251,0.16,0.15/32,0.03,0.173/32,0.036,0.025]

	rdata = b""
	for cr in crs:
		size = int(32*32*3*cr)
		sdata = bytearray(os.urandom(size))
		msg_size = len(sdata)
		sdata = struct.pack(">L", len(sdata))+sdata
		avg_t = 0
		for i in range(10):
			t1 = datetime.datetime.now()
			client.send(sdata)
			while len(rdata) < msg_size:
				tmp_str = client.recv(4096)
				if not tmp_str:break
				rdata += tmp_str
			rdata = rdata[msg_size:]
			rtt = (datetime.datetime.now() - t1).total_seconds()
			avg_t += rtt/2
		avg_t /= 10
		print(cr,avg_t)

	for cr in crs:
		size = int(224*224*3*cr)
		sdata = bytearray(os.urandom(size))
		msg_size = len(sdata)
		sdata = struct.pack(">L", len(sdata))+sdata
		avg_t = 0
		for i in range(10):
			t1 = datetime.datetime.now()
			client.send(sdata)
			while len(rdata) < msg_size:
				tmp_str = client.recv(4096)
				if not tmp_str:break
				rdata += tmp_str
			rdata = rdata[msg_size:]
			rtt = (datetime.datetime.now() - t1).total_seconds()
			avg_t += rtt/2
		avg_t /= 10
		print(cr,avg_t)
	client.close()

if __name__ == "__main__":
	np.random.seed(123)
	torch.manual_seed(2)

	# test 
	deepcod_send()

