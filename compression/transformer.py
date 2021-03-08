import cv2
import numpy as np
import time
import torch
import glob

from torchvision import transforms
from torch.utils.data import Dataset
from collections import OrderedDict
from PIL import Image

dataset = 'ucf101-24'

def get_edge_feature(frame, edge_blur_rad=11, edge_blur_var=0, edge_canny_low=101, edge_canny_high=255):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	start = time.perf_counter()
	blur = cv2.GaussianBlur(gray, (edge_blur_rad, edge_blur_rad), edge_blur_var)
	edge = cv2.Canny(blur, edge_canny_low, edge_canny_high)
	end = time.perf_counter()
	return edge, end-start
    

def get_KAZE_feature(frame):
	alg = cv2.KAZE_create()
	start = time.perf_counter()
	kps = alg.detect(frame)
	end = time.perf_counter()
	kps = sorted(kps, key=lambda x: -x.response)[:32]
	points = [p.pt for p in kps]
	return points, end-start

def get_harris_corner(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	start = time.perf_counter()
	dst = cv2.cornerHarris(gray,2,3,0.04)
	end = time.perf_counter()

	# Threshold for an optimal value, it may vary depending on the image.
	dst[dst>0.01*dst.max()]=[255]
	dst[dst<255]=[0]
	return dst, end-start

def get_GFTT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	start = time.perf_counter()
	corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
	end = time.perf_counter()
	if corners is not None:
		corners = np.int0(corners) 
		points = [i.ravel() for i in corners]
	else:
		points = []
	return points, end-start

# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16
def get_SIFT(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	sift = cv2.xfeatures2d.SIFT_create()
	start = time.perf_counter()
	kps = sift.detect(gray,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_SURF(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	surf = cv2.xfeatures2d.SURF_create()
	start = time.perf_counter()
	kps = surf.detect(gray,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_FAST(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	start = time.perf_counter()
	fast = cv2.FastFeatureDetector_create(threshold=50)
	kps = fast.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_STAR(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	# Initiate STAR detector
	star = cv2.xfeatures2d.StarDetector_create()

	# find the keypoints with STAR
	start = time.perf_counter()
	kps = star.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def get_ORB(frame):
	img = frame.copy()
	gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	orb = cv2.ORB_create()
	start = time.perf_counter()
	kps = orb.detect(img,None)
	end = time.perf_counter()
	points = [p.pt for p in kps]
	return points, end-start

def count_point_in_ROI(ROI, points):
	counter = 0
	for px,py in points:
		inROI = False
		x1,y1,x2,y2 = ROI
		if x1<=px and x2>px and y1<=py and y2>py:
			inROI = True
		if inROI:counter += 1
	return counter

def count_mask_in_ROI(ROI, mp):
	total_pts = np.count_nonzero(mp)
	x1,y1,x2,y2 = ROI
	mp[y1:y2,x1:x2] = 0
	nonroi_pts = np.count_nonzero(mp)
	return total_pts-nonroi_pts

def ROI_area(ROIs,w,h):
	im = np.zeros((h,w),dtype=np.uint8)
	for x1,y1,x2,y2 in ROIs:
		im[y1:y2,x1:x2] = 1
	roi = np.count_nonzero(im)
	return roi

def path_to_disturbed_image(pil_image, label, r_in, r_out):
	b,g,r = cv2.split(np.array(pil_image))
	np_img = cv2.merge((b,g,r))
	np_img = region_disturber(np_img,label, r_in, r_out)
	pil_image = Image.fromarray(np_img)
	return pil_image

# change quality of non-ROI
# r_in is the scaled ratio of ROIs
# r_out is the scaled ratio of the whole image
def region_disturber(image,label,r_in,r_out):
	# get the original content from ROI
	# downsample rest, then upsample
	# put roi back
	w,h = 320,240
	dsize_out = (int(w*r_out),int(h*r_out))
	crops = []
	for _,cx,cy,imgw,imgh  in label:
		cx=int(cx*320);cy=int(cy*240);imgw=int(imgw*320);imgh=int(imgh*320)
		x1=max(cx-imgw//2,0);x2=min(cx+imgw//2,w);y1=max(cy-imgw//2,0);y2=min(cy+imgw//2,h)
		crop = image[y1:y2,x1:x2]
		if r_in<1:
			dsize_in = (int((x2-x1)*r_in),int((y2-y1)*r_in))
			crop_d = cv2.resize(crop, dsize=dsize_in, interpolation=cv2.INTER_LINEAR)
			crop = cv2.resize(crop_d, dsize=(x2-x1,y2-y1), interpolation=cv2.INTER_LINEAR)
		crops.append((x1,y1,x2,y2,crop))
	if r_out<1:
		# downsample
		downsample = cv2.resize(image, dsize=dsize_out, interpolation=cv2.INTER_LINEAR)
		# upsample
		image = cv2.resize(downsample, dsize=(w,h), interpolation=cv2.INTER_LINEAR)
	for x1,y1,x2,y2,crop  in crops:
		image[y1:y2,x1:x2] = crop
	
	return image

# analyze static and motion feature points
# need to count the number of features ROI and not in ROI
# calculate the density
# should compare  
# percentage of features/percentage of area
def analyzer(image):
	# analyze features in image
	bgr_frame = np.array(image)
	# FAST
	fast, _ = get_FAST(bgr_frame)
	# STAR
	star, _ = get_STAR(bgr_frame)
	# ORB
	orb, _ = get_ORB(bgr_frame)

class LRU(OrderedDict):

	def __init__(self, maxsize=128):
		self.maxsize = maxsize
		super().__init__()

	def __getitem__(self, key):
		value = super().__getitem__(key)
		self.move_to_end(key)
		return value

	def __setitem__(self, key, value):
		if key in self:
			self.move_to_end(key)
		super().__setitem__(key, value)
		if len(self) > self.maxsize:
			oldest = next(iter(self))
			del self[oldest]

def tile_disturber(image, C_param):
	# analyze features in image
	feat_start = time.perf_counter()
	bgr_frame = np.array(image)
	# edge diff
	# edge, _ = get_edge_feature(bgr_frame)
	# harris corner
	hc, _ = get_harris_corner(bgr_frame)
	# # GFTT
	# gftt, _ = get_GFTT(bgr_frame)
	# FAST
	fast, _ = get_FAST(bgr_frame)
	# STAR
	star, _ = get_STAR(bgr_frame)
	# ORB
	orb, _ = get_ORB(bgr_frame)

	calc_start = time.perf_counter()
	point_features = [fast, star, orb]
	map_features = []
	num_features = len(point_features) + len(map_features)
	# divide image to 4*3 tiles
	ROIs = []
	num_w, num_h = 4,3
	img_h,img_w = bgr_frame.shape[:2]
	tilew,tileh = img_w//num_w,img_h//num_h
	if img_w%num_w != 0:tilew += 1
	if img_h%num_h != 0:tileh += 1
	for row in range(num_h):
		for col in range(num_w):
			x1 = col*tilew; x2 = min((col+1)*tilew,img_w); y1 = row*tileh; y2 = min((row+1)*tileh,img_h)
			ROIs.append([x1,y1,x2,y2])
	counts = np.zeros((num_w*num_h,num_features))
	for roi_idx,ROI in enumerate(ROIs):
		roi_start = time.perf_counter()
		feat_idx = 0
		for mf in map_features:
			c = count_mask_in_ROI(ROI,mf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		for pf in point_features:
			c = count_point_in_ROI(ROI,pf)
			counts[roi_idx,feat_idx] = c
			feat_idx += 1
		roi_end = time.perf_counter()

	# weight of different features
	weights = C_param[:num_features] + 0.5
	# lower and upper
	lower,upper = C_param[num_features:num_features+2] + 0.5
	lower,upper = min(lower,upper),max(lower,upper)
	lower = max(lower,0);upper = min(upper,1)
	# order to adjust the concentration of the  scores
	k = int(((C_param[num_features+2]+0.5)*5))
	k = min(k,4); k = max(k,0)
	order_choices = [1./3,1./2,1,2,3]
	# score of each feature sum to 1
	normalized_score = counts/(np.sum(counts,axis=0)+1e-6)
	weights /= (np.sum(weights)+1e-6)
	# ws of all tiles sum up to 1
	weighted_scores = np.matmul(normalized_score,weights)
	# the weight is more valuable when its value is higher
	quality = (upper-lower)*weighted_scores**order_choices[k] + lower

	tile_sizes = [(int(np.rint((x2-x1)*r)),int(np.rint((y2-y1)*r))) for r,(x1,y1,x2,y2) in zip(quality,ROIs)]

	# not used for training,but can be used for 
	# ploting the pareto front
	compressed_size = 0
	tile_size = tilew * tileh
	for roi,dsize in zip(ROIs,tile_sizes):
		x1,y1,x2,y2 = roi
		if dsize == (x2-x1,y2-y1):
			compressed_size += (x2-x1)*(y2-y1)
			continue
		crop = bgr_frame[y1:y2,x1:x2].copy()
		if dsize[0]==0 or dsize[1]==0:
			bgr_frame[y1:y2,x1:x2] = [0]
		else:
			try:
				crop_d = cv2.resize(crop, dsize=dsize, interpolation=cv2.INTER_LINEAR)
				crop = cv2.resize(crop_d, dsize=(x2-x1,y2-y1), interpolation=cv2.INTER_LINEAR)
			except Exception as e:
				print(repr(e))
				print(C_param,tile_sizes)
				print('dsize:',dsize,crop.shape)
				exit(1)
			compressed_size += dsize[0]*dsize[1]
			bgr_frame[y1:y2,x1:x2] = crop

	feat_end = time.perf_counter()
	# print(img_index,feat_end-feat_start)
	return bgr_frame,compressed_size

def JPEG_disturber(image, C_param):
	return image

# define a class for transformation
class Transformer:
	def __init__(self,name):
		# need a dict as buffer to store transformed image of a range
		self.name = name

	def transform(self, image=None, C_param=None):
		self.original_size += image.shape[0]*image.shape[1]
		rimage,comp_sz = tile_disturber(image, C_param)
		self.compressed_size += comp_sz
		return rimage

	def reset(self):
		self.compressed_size = 0
		self.original_size = 0

	def get_compression_ratio(self):
		assert(self.original_size>0)
		return 1-1.0*self.compressed_size/self.original_size

if __name__ == "__main__":
    # img = cv2.imread('/home/bo/research/dataset/ucf24/compressed/000000.jpg')
    img = cv2.imread('/home/bo/research/dataset/ucf24/rgb-images/Basketball/v_Basketball_g01_c01/00001.jpg')
    analyzer(img)