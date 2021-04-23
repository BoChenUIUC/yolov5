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

no_of_hidden_units = 196
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.conv1 = nn.Conv2d(3, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		# self.ln1 = nn.LayerNorm([no_of_hidden_units,32,32])
		self.bn1 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu1 = nn.LeakyReLU()

		self.conv2 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		# self.ln2 = nn.LayerNorm([no_of_hidden_units,16,16])
		self.bn2 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu2 = nn.LeakyReLU()

		self.conv3 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		# self.ln3 = nn.LayerNorm([no_of_hidden_units,16,16])
		self.bn3 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu3 = nn.LeakyReLU()

		self.conv4 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		# self.ln4 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.bn4 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu4 = nn.LeakyReLU()

		self.conv5 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		# self.ln5 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.bn5 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu5 = nn.LeakyReLU()

		self.conv6 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		# self.ln6 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.bn6 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu6 = nn.LeakyReLU()

		self.conv7 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=1, padding=1)
		# self.ln7 = nn.LayerNorm([no_of_hidden_units,8,8])
		self.bn7 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu7 = nn.LeakyReLU()

		self.conv8 = nn.Conv2d(no_of_hidden_units, no_of_hidden_units, kernel_size=3, stride=2, padding=1)
		# self.ln8 = nn.LayerNorm([no_of_hidden_units,4,4])
		self.bn8 = nn.BatchNorm2d(no_of_hidden_units, momentum=0.01, eps=1e-3)
		self.lrelu8 = nn.LeakyReLU()

		self.adaptive_pool = nn.AdaptiveAvgPool2d(4)

		self.pool = nn.MaxPool2d(4, 4)
		self.fc1 = nn.Linear(no_of_hidden_units, 1)

	def forward(self, x, extract_features=0):
		x = self.bn1(self.lrelu1(self.conv1(x)))
		x = self.bn2(self.lrelu2(self.conv2(x)))
		x = self.bn3(self.lrelu3(self.conv3(x)))
		x = self.bn4(self.lrelu4(self.conv4(x)))
		x = self.bn5(self.lrelu5(self.conv5(x)))
		x = self.bn6(self.lrelu6(self.conv6(x)))
		x = self.bn7(self.lrelu7(self.conv7(x)))
		x = self.bn8(self.lrelu8(self.conv8(x)))
		x = self.adaptive_pool(x)
		x = self.pool(x)
		x = x.view(-1, no_of_hidden_units)
		y1 = self.fc1(x)
		return y1

def compute_gradient_penalty(D, real_samples, fake_samples, cuda):
	"""Calculates the gradient penalty loss for WGAN GP"""
	# Random weight term for interpolation between real and fake samples
	alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
	if cuda:alpha = alpha.cuda()
	# Get random interpolation between real and fake samples
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = D(interpolates)
	fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
	if cuda: fake = fake.cuda()
	# Get gradient w.r.t. interpolates
	gradients = torch.autograd.grad(
		outputs=d_interpolates,
		inputs=interpolates,
		grad_outputs=fake,
		create_graph=True,
		retain_graph=True,
		only_inputs=True,
	)[0]
	gradients = gradients.view(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty

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

		# quantization
		xsize = list(x.size())
		x = x.view(*(xsize + [1]))
		quant_dist = torch.pow(x-self.centers, 2)
		softout = torch.sum(self.centers * nn.functional.softmax(-quant_dist, dim=-1), dim=-1)
		maxval = torch.min(quant_dist, dim=-1, keepdim=True)[0]
		hardout = torch.sum(self.centers * (maxval == quant_dist), dim=-1)
		# dont know how to use hardout, use this temporarily
		x = softout

		return x

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
		self.encoder = LightweightEncoder(out_size, kernel_size=4, num_centers=8)
		self.conv1 = Middle_conv(out_size,out_size)
		self.resblock_up1 = Resblock_up(out_size,no_of_hidden_units)
		self.conv2 = Middle_conv(no_of_hidden_units,no_of_hidden_units)
		self.resblock_up2 = Resblock_up(no_of_hidden_units,no_of_hidden_units)
		self.output_conv = Output_conv(no_of_hidden_units)
		

	def forward(self, x): 
		x = self.encoder(x)

		# reconstruct
		x = self.conv1(x)
		x = self.resblock_up1(x)
		x = self.conv2(x)
		x = self.resblock_up2(x)
		x = self.output_conv(x)
		
		return x

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