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

# eceived: 587 4066 -0.038248  3.89
# Received: 737 3449 -0.038131 4.02
# Received: 3612 6778 -0.038024 4.775
# Received: 5419 7232 -0.037943
# Received: 24987 24987 -0.037829

# 11.676-11.213
# 27.6337-27.7133
time_offset = 0
def deepcod_recv():
	import datetime
	payload_size = struct.calcsize(">L")
	serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	ADDR = ("130.126.136.154",8848)
	# ADDR = ("127.0.0.1",8848)
	serv.bind(ADDR)
	serv.listen(5)
	print('Waiting',ADDR)
	conn, addr = serv.accept()
	# connect to edge
	print('Connected.')
	data = b""
	cnt,total = 0,0
	while True:
		# decode time
		while len(data) < 26:
			data += conn.recv(4096)
		edge_send_time = datetime.datetime.strptime(data[:26].decode(), "%Y-%m-%d %H:%M:%S.%f") - datetime.timedelta(seconds=time_offset)
		data = data[26:]
		while len(data) < payload_size:
			data += conn.recv(4096)
		msg_size = struct.unpack(">L", data[:payload_size])[0]
		data = data[payload_size:]
		while len(data) < msg_size:
			tmp_str = conn.recv(4096)
			if not tmp_str:break
			data += tmp_str
		cloud_recv_time = datetime.datetime.now()
		diff = (cloud_recv_time - edge_send_time).total_seconds()
		print('Received:',msg_size,len(data),diff)
		cnt += 1
		total += diff
		if cnt == 10:
			print(msg_size,total/10.0)
			cnt = 0
			total = 0
		data = data[msg_size:]
	# while True:
	# 	while len(data) < payload_size:
	# 		data += conn.recv(4096)
	# 	msg_size = struct.unpack(">L", data[:payload_size])[0]
	# 	data = data[payload_size:]
	# 	while len(data) < msg_size:
	# 		tmp_str = conn.recv(4096)
	# 		if not tmp_str:break
	# 		data += tmp_str
	# 	print('Received:',msg_size,len(data))

	# 	conn.send(data[:msg_size])
	# 	data = data[msg_size:]

if __name__ == "__main__":
	np.random.seed(123)
	torch.manual_seed(2)

	deepcod_recv()
