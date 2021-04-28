import cv2
import numpy as np
import time
import torch
import glob
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.autograd import Variable
from torch.nn.utils import spectral_norm
from huffman import HuffmanCoding

def orthorgonal_regularizer(w,scale,cuda=False):
	N, C, H, W = w.size()
	w = w.view(N*C, H, W)
	weight_squared = torch.bmm(w, w.permute(0, 2, 1))
	ones = torch.ones(N * C, H, H, dtype=torch.float32)
	diag = torch.eye(H, dtype=torch.float32)
	tmp = ones - diag
	if cuda:tmp = tmp.cuda()
	loss_orth = ((weight_squared * tmp) ** 2).sum()
	return loss_orth*scale

class Resblock_up(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Resblock_up, self).__init__()
		self.bn1 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		deconv1 = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv1 = spectral_norm(deconv1)

		self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.01, eps=1e-3)
		deconv2 = nn.ConvTranspose2d(out_channels, out_channels, 3, stride=1, padding=1)
		self.deconv2 = spectral_norm(deconv2)

		self.bn3 = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		deconv_skip = nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1)
		self.deconv_skip = spectral_norm(deconv_skip)

	def forward(self, x_init):
		x = self.deconv1(F.relu(self.bn1(x_init)))
		x = self.deconv2(F.relu(self.bn2(x)))
		x_init = self.deconv_skip(F.relu(self.bn3(x_init)))
		return x + x_init

class Middle_conv(nn.Module):

	def __init__(self, in_channels, out_channels):
		super(Middle_conv, self).__init__()
		self.bn = nn.BatchNorm2d(in_channels, momentum=0.01, eps=1e-3)
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv = spectral_norm(self.conv)

	def forward(self, x):
		x = self.conv(F.relu(self.bn(x)))

		return x


class LightweightEncoder(nn.Module):

	def __init__(self, channels, kernel_size=4, num_centers=8):
		super(LightweightEncoder, self).__init__()
		self.sample = nn.Conv2d(3, channels, kernel_size=kernel_size, stride=kernel_size, padding=0, bias=True)
		self.sample = spectral_norm(self.sample)
		self.centers = torch.nn.Parameter(torch.rand(num_centers))

	def forward(self, x):
		# sample from input
		x = self.sample(x)
		B,C,H,W = x.size()

		# quantization
		xsize = list(x.size())
		x = x.view(*(xsize + [1]))
		quant_dist = torch.pow(x-self.centers, 2)
		softout = torch.sum(self.centers * nn.functional.softmax(-quant_dist, dim=-1), dim=-1)
		minval,index = torch.min(quant_dist, dim=-1, keepdim=True)
		hardout = torch.sum(self.centers * (minval == quant_dist), dim=-1)
		# dont know how to use hardout, use this temporarily
		x = softout

		huffman = HuffmanCoding()
		real_size = len(huffman.compress(index.view(-1).cpu().numpy())) * 4
		real_cr = 1/16.*real_size/(H*W*C*B*8)
		return x,real_cr

class Output_conv(nn.Module):

	def __init__(self, channels):
		super(Output_conv, self).__init__()
		self.bn = nn.BatchNorm2d(channels, momentum=0.01, eps=1e-3)
		# self.relu = nn.LeakyReLU()#nn.ReLU(inplace=True)
		self.conv = nn.Conv2d(channels, 3, kernel_size=3, stride=1, padding=1, bias=True)
		self.conv = spectral_norm(self.conv)

	def forward(self, x):
		x = self.conv(F.relu(self.bn(x)))
		x = torch.tanh(x)
		x = (x+1)/2

		return x

def init_weights(m):
	if isinstance(m, nn.Conv2d):
		nn.init.kaiming_normal_(m.weight, mode='fan_out')
		nn.init.constant_(m.bias, 0)


class DeepCOD(nn.Module):

	def __init__(self, kernel_size=4, num_centers=8):
		super(DeepCOD, self).__init__()
		out_size = 3
		no_of_hidden_units = 64
		self.encoder = LightweightEncoder(out_size, kernel_size=4, num_centers=8)
		self.conv1 = Middle_conv(out_size,out_size)
		self.resblock_up1 = Resblock_up(out_size,no_of_hidden_units)
		self.conv2 = Middle_conv(no_of_hidden_units,no_of_hidden_units)
		self.resblock_up2 = Resblock_up(no_of_hidden_units,no_of_hidden_units)
		self.output_conv = Output_conv(no_of_hidden_units)
		

	def forward(self, x): 
		x,r = self.encoder(x)

		# reconstruct
		x = self.conv1(x)
		x = self.resblock_up1(x)
		x = self.conv2(x)
		x = self.resblock_up2(x)
		x = self.output_conv(x)
		
		return x,r

if __name__ == '__main__':
	image = torch.randn(1,3,32,32)
	model = DeepCOD()
	output = model(image)
	print(model)
	# print(output.shape)
	# weight = torch.diag(torch.ones(4)).repeat(3,3,1,1)
	# print(weight.size())
	print(model.encoder.sample.weight.size())
	# r = orthorgonal_regularizer(model.sample.weight,1,False)
	# print(r)
	# for name, param in model.named_parameters():
	# 	print('name is {}'.format(name))
	# 	print('shape is {}'.format(param.shape))