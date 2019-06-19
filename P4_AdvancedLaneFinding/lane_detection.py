import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("-i","--image",help="input image file path")
args=parser.parse_args()

def display(window_name,img,pts=[]):
	copy=img.copy()
	if len(copy.shape)<3:
		copy=np.expand_dims(copy,axis=2)
		copy=np.concatenate([copy,copy,copy],axis=2)
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

def transform_view_given_pts(img,src_pts,dst_pts):
	P=cv2.getPerspectiveTransform(np.array(src_pts,dtype=np.float32),np.array(dst_pts,dtype=np.float32))
	return cv2.warpPerspective(img,P,img.shape[0:2][::-1]),P

def transform_view_given_projection(img,projection_matrix):
	return cv2.warpPerspective(img,projection_matrix,img.shape[0:2][::-1]),projection_matrix

def transform_view(img,*args):
	if len(args)==1:
		return transform_view_given_projection(img,args[0])
	if len(args)==2:
		return transform_view_given_pts(img,args[0],args[1])

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

def find_lanes(lanes_image,debug_mode=True):
	nrows,ncols=lanes_image.shape
	hist=np.sum(lanes_image[nrows//2:nrows,:],axis=0)
	if debug_mode:
		plt.plot(hist)
		plt.show()

	leftx=np.argmax(hist[0:ncols//2])
	rightx=np.argmax(hist[ncols//2:ncols])+(ncols//2)

	nwindows=9
	window_h=nrows/nwindows
	window_w=100
	minpix=50

	copy=lanes_image.copy()
	copy=np.expand_dims(copy,axis=2)
	copy=np.concatenate([copy,copy,copy],axis=2)

	mask_out=np.zeros_like(copy)

	probable_ids=lanes_image.nonzero()
	left_inds_list=[]
	right_inds_list=[]
	for window in range(nwindows):
		window_ytop=nrows-(window+1)*window_h
		window_ybottom=nrows-(window)*window_h
		window_xleft=leftx-window_w
		window_xright=leftx+window_w
		cv2.rectangle(copy,(int(window_xleft),int(window_ytop)),(int(window_xright),int(window_ybottom)),(255,0,0),3)
		left_inds=((probable_ids[0]>=window_ytop) & (probable_ids[0]<=window_ybottom)
					& (probable_ids[1]>=window_xleft) & (probable_ids[1]<=window_xright)).nonzero()[0]
		left_inds_list.append(left_inds)

		window_xleft=rightx-window_w
		window_xright=rightx+window_w
		cv2.rectangle(copy,(int(window_xleft),int(window_ytop)),(int(window_xright),int(window_ybottom)),(0,0,255),3)
		right_inds=((probable_ids[0]>=window_ytop) & (probable_ids[0]<=window_ybottom)
					& (probable_ids[1]>=window_xleft) & (probable_ids[1]<=window_xright)).nonzero()[0]
		right_inds_list.append(right_inds)
		
		if len(left_inds)>minpix:
			leftx=np.int(np.mean(probable_ids[1][left_inds]))
		if len(right_inds)>minpix:
			rightx=np.int(np.mean(probable_ids[1][right_inds]))

	left_inds_list=np.concatenate(left_inds_list)
	right_inds_list=np.concatenate(right_inds_list)

	left_xpts=probable_ids[1][left_inds_list]
	left_ypts=probable_ids[0][left_inds_list]
	left_poly=np.zeros((4,),dtype=np.float32)
	left_lane_found=False
	if left_inds_list.shape[0]>=minpix:
		left_poly=np.polyfit(left_ypts,left_xpts,3)
		left_lane_found=True
		y_sample=np.linspace(0,nrows-1,nrows)
		left_xtest=left_poly[0]*y_sample**3+left_poly[1]*y_sample**2+left_poly[2]*y_sample+left_poly[3]
		lane_pts=np.int_(np.vstack([left_xtest,y_sample]))
		lane_pts=np.transpose(lane_pts)
		lane_pts=lane_pts.reshape((-1,1,2))
		cv2.polylines(copy,[lane_pts],False,(255,0,0),3)
		cv2.polylines(mask_out,[lane_pts],False,(0,255,0),25)

	right_xpts=probable_ids[1][right_inds_list]
	right_ypts=probable_ids[0][right_inds_list]
	right_poly=np.zeros((4,),dtype=np.float32)
	right_lane_found=False
	if right_inds_list.shape[0]>=minpix:
		right_poly=np.polyfit(right_ypts,right_xpts,3)
		right_lane_found=True
		y_sample=np.linspace(0,nrows-1,nrows)
		right_xtest=right_poly[0]*y_sample**3+right_poly[1]*y_sample**2+right_poly[2]*y_sample+right_poly[3]
		lane_pts=np.int_(np.vstack([right_xtest,y_sample]))
		lane_pts=np.transpose(lane_pts)
		lane_pts=lane_pts.reshape((-1,1,2))
		cv2.polylines(copy,[lane_pts],False,(0,0,255),3)
		cv2.polylines(mask_out,[lane_pts],False,(0,255,0),25)

	if debug_mode:
		display("detected lane mark polynomials",copy)

	return [left_poly,left_lane_found,right_poly,right_lane_found,mask_out]

def draw_lanes(image,mask,src_pts,dst_pts):
	display("mask",mask)
	unwarped_mask,_=transform_view(mask,src_pts,dst_pts)
	unwarped_mask=unwarped_mask.astype(np.uint8)
	display("unwarped mask",unwarped_mask)
	image=cv2.addWeighted(image,1,unwarped_mask,0.8,0)
	display("lanes on original image",image)

imfile=args.image

src_pts=[(580,460),(205,720),(1110,720),(703,460)]
dst_pts=[(320,0),(320,720),(960,720),(960,0)]

img=cv2.imread(imfile)
display("image",img,pts=src_pts)

warped_img,P=transform_view(img,src_pts,dst_pts)
display("warped image",warped_img,pts=dst_pts)

filtered_img=gaussian_smoothing(warped_img,ksize=(3,3))
display("filtered image",filtered_img)

hls_img=cv2.cvtColor(filtered_img,cv2.COLOR_BGR2HLS)
display("hls image",hls_img)

params={'s_thresh_y':120,'l_thresh_y':40,'l_thresh_w':205,
'orientation_thresh_min':0.0,'orientation_thresh_max':0.5,'magnitude_thresh':80}
lane_mask=lane_thresholding(hls_img,params)
display("lane mask",lane_mask)

left_poly,left_lane_found,right_poly,right_lane_found,lane_mask=find_lanes(lane_mask,debug_mode=True) 

draw_lanes(img.copy(),lane_mask,dst_pts,src_pts)