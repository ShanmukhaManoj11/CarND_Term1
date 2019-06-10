import numpy as np
import cv2
import os

def display(window_name,img,pts=[]):
	copy=img.copy()
	if len(pts)!=0:
		for p in pts:
			cv2.circle(copy,p,5,(255,0,0),-1)
		for i in range(len(pts)):
			cv2.line(copy,pts[i],pts[(i+1)%len(pts)],(255,255,255),2)
	cv2.imshow(window_name,copy)
	while cv2.waitKey(1) != ord("q"):
		continue
	cv2.destroyAllWindows()

def crop(img,xi_offset=0,xf_offset=0,yi_offset=0,yf_offset=0):
	#(0,0) is top left corner and x is positve to right and y is positive down
	#pixel at (x,y) is accessed as img[y,x]
	nrows,ncols=img.shape[0:2]
	img=img[yi_offset:nrows-yf_offset,xi_offset:ncols-xf_offset]
	return img

def gaussian_smoothing(img,ksize=(3,3)):
	return cv2.GaussianBlur(img,ksize,0)

def transform_view(img,src_pts,dst_pts):
	P=cv2.getPerspectiveTransform(np.array(src_pts,dtype=np.float32),np.array(dst_pts,dtype=np.float32))
	return cv2.warpPerspective(img,P,img.shape[0:2][::-1])

def ls_thresholding(hls,s_thresh_y=120,l_thresh_y=40,l_thresh_w=205):
	l=hls[:,:,1].astype(np.float)
	s=hls[:,:,2].astype(np.float)
	cond1=(s>s_thresh_y) & (l>l_thresh_y)
	cond2=(l>l_thresh_w)
	b=np.zeros_like(s)
	b[cond1 | cond2]=1
	return b

def gradient_thresholding(hls,orientation_thresh_min=0,orientation_thresh_max=0.7,magnitude_thresh=40):
	l=hls[:,:,1].astype(np.float64)
	lx=cv2.Sobel(l,cv2.CV_64F,1,0,ksize=5)
	ly=cv2.Sobel(l,cv2.CV_64F,0,1,ksize=5)
	orientation=np.arctan2(np.absolute(ly),np.absolute(lx))
	mag=np.sqrt(np.square(lx),np.square(ly))
	scaled_mag=255.0*np.absolute(mag)/np.max(np.absolute(mag))
	b=np.zeros_like(l)
	cond1=(scaled_mag>magnitude_thresh)
	cond2=(orientation>orientation_thresh_min) & (orientation<orientation_thresh_max)
	b[cond1 & cond2]=1
	return b

def lane_thresholding(hls,params):
	b1=ls_thresholding(hls,s_thresh_y=params['s_thresh_y'],l_thresh_y=params['l_thresh_y'],
		l_thresh_w=params['l_thresh_w'])
	b2=gradient_thresholding(hls,orientation_thresh_min=params['orientation_thresh_min'],
		orientation_thresh_max=params['orientation_thresh_max'],magnitude_thresh=params['magnitude_thresh'])
	return cv2.bitwise_or(b1,b2)

base_path="./test_images/"
test_imfile="test1.jpg"

src_pts=[(580,460),(205,720),(1110,720),(703,460)]
dst_pts=[(320,0),(320,720),(960,720),(960,0)]

img=cv2.imread(base_path+test_imfile)
display("image",img,pts=src_pts)

warped_img=transform_view(img,src_pts,dst_pts)
display("warped image",warped_img,pts=dst_pts)

filtered_img=gaussian_smoothing(warped_img,ksize=(3,3))
display("filtered image",filtered_img)

hls_img=cv2.cvtColor(filtered_img,cv2.COLOR_BGR2HLS)
display("hls image",hls_img)

params={'s_thresh_y':120,'l_thresh_y':40,'l_thresh_w':205,
'orientation_thresh_min':0.0,'orientation_thresh_max':0.5,'magnitude_thresh':40}
lane_mask=lane_thresholding(hls_img,params)
display("lane mask",lane_mask)