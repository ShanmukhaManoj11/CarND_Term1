import numpy as np
import cv2
import math
import matplotlib.image as mpimg
from skimage.feature import hog
from scipy.ndimage.measurements import label

def bin_spatial(img,size=(32,32)):
    features=cv2.resize(img,size).ravel() 
    return features

def color_hist(img,nbins=32):
    channel1_hist=np.histogram(img[:,:,0],bins=nbins)
    channel2_hist=np.histogram(img[:,:,1],bins=nbins)
    channel3_hist=np.histogram(img[:,:,2],bins=nbins)
    hist_features=np.concatenate((channel1_hist[0],channel2_hist[0],channel3_hist[0]))
    return hist_features

def get_hog_features(img,orient,pix_per_cell,cell_per_block,vis=False,feature_vec=True):
    if vis == True:
        features,hog_image=hog(img,orientations=orient,
                               pixels_per_cell=(pix_per_cell,pix_per_cell),
                               cells_per_block=(cell_per_block,cell_per_block),
                               transform_sqrt=False,
                               visualise=vis,feature_vector=feature_vec)
        return features, hog_image
    else:      
        features=hog(img,orientations=orient,
                     pixels_per_cell=(pix_per_cell,pix_per_cell),
                     cells_per_block=(cell_per_block,cell_per_block),
                     transform_sqrt=False,
                     visualise=vis,feature_vector=feature_vec)
        return features
    
def single_img_features(image,cspace='RGB',spatial_size=(32,32),hist_bins=32,orient=9,
                        pix_per_cell=8,cell_per_block=2,hog_channel="ALL"):    
    image_features=[]
    if cspace!='RGB':
        if cspace=='HSV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        elif cspace=='LUV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
        elif cspace=='HLS':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        elif cspace=='YUV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
        elif cspace=='YCrCb':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
    else:
        feature_image=np.copy(img)   

    spatial_features=bin_spatial(feature_image,size=spatial_size)
    image_features.append(spatial_features)

    hist_features=color_hist(feature_image,nbins=hist_bins)
    image_features.append(hist_features)

    if hog_channel=='ALL':
        hog_features=[]
        for channel in range(feature_image.shape[2]):
            hog_features.append(get_hog_features(feature_image[:,:,channel],orient,pix_per_cell,cell_per_block,
                                                 vis=False,feature_vec=True))
        hog_features=np.concatenate(hog_features)        
    else:
        hog_features=get_hog_features(feature_image[:,:,hog_channel],orient,pix_per_cell,cell_per_block,
                                      vis=False,feature_vec=True)
    image_features.append(hog_features)

    return np.concatenate(image_features)

def extract_features(imgs,cspace='RGB',spatial_size=(32,32),hist_bins=32,
                     orient=9,pix_per_cell=8,cell_per_block=2,hog_channel="ALL"):
    features=[]
    for file in imgs:
        image=mpimg.imread(file)
        image_features=single_img_features(image,cspace=cspace,spatial_size=spatial_size,hist_bins=hist_bins,
                                           orient=orient,pix_per_cell=pix_per_cell,cell_per_block=cell_per_block,hog_channel=hog_channel)
        features.append(image_features)
        
    return features

def slide_window_find_cars(img,svc,scaler,x_start_stop=[None,None],y_start_stop=[None,None],xy_window=(64,64),xy_overlap=(0.5, 0.5),
                           cspace='RGB',spatial_size=(32,32),hist_bins=32,orient=9,pix_per_cell=8,cell_per_block=2,hog_channel="ALL"):
    if x_start_stop[0]==None:
        x_start_stop[0]=0
    if x_start_stop[1]==None:
        x_start_stop[1]=img.shape[1]
    if y_start_stop[0]==None:
        y_start_stop[0]=0
    if y_start_stop[1]==None:
        y_start_stop[1]=img.shape[0]
    # Compute the span of the region to be searched 
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
    ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
    nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
    ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
    # Initialize a list to append window positions to
    window_list = []
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            img_patch=cv2.resize(img[starty:endy,startx:endx],(64,64))
            patch_features=single_img_features(img_patch,cspace=cspace,spatial_size=spatial_size,hist_bins=hist_bins,
                                               orient=orient,pix_per_cell=pix_per_cell,
                                               cell_per_block=cell_per_block,hog_channel=hog_channel)
            scaled_features=scaler.transform(np.array(patch_features).reshape(1, -1))
            prediction=svc.predict(scaled_features)
            
            if prediction==1:
                # Append window position to list
                window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def find_cars(img,cspace='RGB',scale=1,ystart=None,ystop=None,svc=None,X_scaler=None,
              orient=9,pix_per_cell=8,cell_per_block=2,spatial_size=(32,32),hist_bins=32):

    bboxes=[]
    img=img.astype(np.float32)/255

    if svc==None or X_scaler==None:
        return bboxes

    if ystart==None:
        ystart=0
    if ystop==None:
        ystop==img.shape[0]
    
    image=img[ystart:ystop,:,:]
    if cspace!='RGB':
        if cspace=='HSV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        elif cspace=='LUV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2LUV)
        elif cspace=='HLS':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
        elif cspace=='YUV':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2YUV)
        elif cspace=='YCrCb':
            feature_image=cv2.cvtColor(image,cv2.COLOR_RGB2YCrCb)
    else:
        feature_image=np.copy(image)

    if scale != 1:
        imshape = feature_image.shape
        feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = feature_image[:,:,0]
    ch2 = feature_image[:,:,1]
    ch3 = feature_image[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    
            #test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))    
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                bbox=((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart))
                bboxes.append(bbox)
                
    return bboxes

def draw_boxes(img,bboxes):
    draw_img=np.copy(img)
    if len(bboxes)==0:
        return draw_img
    for bbox in bboxes:
        cv2.rectangle(draw_img,bbox[0],bbox[1],(0,0,255),6)
    return draw_img

def draw_boxes_thresholded(img,bboxes,hist_bboxes,hist_thresh=10,threshold=2):
    draw_img=np.copy(img)
    
    #if len(bboxes)==0:
        #cv2.putText(draw_img,'cars found = 0',(30,90),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        #return draw_img
    
    heatmap=np.zeros_like(img[:,:,0])
    for bbox in bboxes:
        hist_bboxes.append(bbox)
    hist_bboxes=hist_bboxes[-(hist_thresh+len(bboxes)):]
    for bbox in hist_bboxes:
        if len(bbox)!=0:
            heatmap[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]]+=1
    
    heatmap[heatmap<=threshold]=0     
    labels=label(heatmap)
    cv2.putText(draw_img,'cars found = '+str(labels[1]),(30,90),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    for car_number in range(1,labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(draw_img, bbox[0], bbox[1], (0,0,255), 6)
        
    return draw_img,hist_bboxes
