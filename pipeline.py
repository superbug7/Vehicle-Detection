import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from helper_func import *
import pickle
import random
%matplotlib inline
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from moviepy.editor import VideoFileClip
from IPython.display import HTML
#from sklearn.cross_validation import train_test_split
%matplotlib inline
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def augment_image(img):
   new_img = cv2.GaussianBlur(img, (3,3), 0)
   #new_img = cv2.cvtColor(new_img, cv2.COLOR_YUV2RGB)
   new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2HSV)
   new_img = np.array(new_img, dtype = np.float64)
   #Generate new random brightness
   random_bright = .5+random.uniform(0.3,1.0)
   new_img[:,:,2] = random_bright*new_img[:,:,2]
   new_img[:,:,2][new_img[:,:,2]>255]  = 255
   new_img = np.array(new_img, dtype = np.uint8)
    #Convert back to RGB colorspace
   new_img = cv2.cvtColor(new_img, cv2.COLOR_HSV2RGB)
   #new_img = cv2.cvtColor(new_img, cv2.COLOR_RGB2YUV)
   return new_img




# Read in cars and notcars
images = glob.glob('./dataset/*.png')
cars = []
notcars = []
for image in images:
    cars.append(image)
        
images = glob.glob('./dataset_nonv/*.png')
for image in images:
    notcars.append(image)


# Reduce the sample size because
# The quiz evaluator times out after 13s of CPU time
#sample_size = 500
#cars = cars[0:sample_size]
#notcars = notcars[0:sample_size]

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 32  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()


def train_model(cars, notcars):
  car_features = extract_features(cars, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)
  notcar_features = extract_features(notcars, color_space=color_space, 
        spatial_size=spatial_size, hist_bins=hist_bins, 
        orient=orient, pix_per_cell=pix_per_cell, 
        cell_per_block=cell_per_block, 
        hog_channel=hog_channel, spatial_feat=spatial_feat, 
        hist_feat=hist_feat, hog_feat=hog_feat)

  X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
  # Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
  # Apply the scaler to X
  scaled_X = X_scaler.transform(X)

  # Define the labels vector
  y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


  # Split up data into randomized training and test sets
  rand_state = np.random.randint(0, 100)
  X_train, X_test, y_train, y_test = train_test_split(
      scaled_X, y, test_size=0.2, random_state=rand_state)

  print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
  print('Feature vector length:', len(X_train[0]))
  # Use a linear SVC 
  svc = LinearSVC()
  # Check the training time for the SVC
  t=time.time()
  svc.fit(X_train, y_train)
  t2 = time.time()
  print(round(t2-t, 2), 'Seconds to train SVC...')
  # Check the score of the SVC
  print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
  # Check the prediction time for a single sample
  t=time.time()
  #model = pickle.dump(svc, 'model.pkl')
  with open('model.p', 'wb') as f:
        pickle.dump((svc, X_scaler), f)

  #return svc, X_scaler
  #image = mpimg.imread('test1.jpg')
  #draw_image = np.copy(image)

# Uncomment the following line if you extracted training
# data from .png images (scaled 0 to 1 by mpimg) and the
# image you are searching is a .jpg (scaled 0 to 255)
#image = image.astype(np.float32)/255

#windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop, 
#                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

#hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space, 
#                        spatial_size=spatial_size, hist_bins=hist_bins, 
#                        orient=orient, pix_per_cell=pix_per_cell, 
#                        cell_per_block=cell_per_block, 
#                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                        hist_feat=hist_feat, hog_feat=hog_feat)                       

#window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)                    

#plt.imshow(window_img)


def find_vehicles_in_frame(image):
  ystart = 400
  ystop = 656
  scale = 1.5
  box_list1 = []
  box_list2 = []
  #image = mpimg.imread('test1.jpg')
  svc, X_scaler = pickle.load( open("model.p", "rb" ) )
  box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 400, 464, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 416, 480, 1, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 400, 500, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 430, 530, 1.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 400, 530, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 430, 560, 2, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 400, 600, 3.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  box_list += find_cars(image, 464, 656, 3.5, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)


  #ystart = 355
  #ystop = 550
  #scale = 1.5
  #box_list2 = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  #box_list = box_list1 + box_list2

  heat = np.zeros_like(image[:,:,0]).astype(np.float)

  # Add heat to each box in box list
  heat = add_heat(heat,box_list)

  # Apply threshold to help remove false positives
  heat = apply_threshold(heat,1)

  # Visualize the heatmap when displaying
  heatmap = np.clip(heat, 0, 255)

  # Find final boxes from heatmap using label function
  labels = label(heatmap)
  #return labels, heatmap
  draw_img = draw_labeled_bboxes(np.copy(image), labels)
  return draw_img


def find_vehicles_in_video(video):
    output = "tracked2_" + video
    input_clip = VideoFileClip(video)
    clip = input_clip.fl_image(find_vehicles_in_frame)
    #clip = input_clip.fl_image(save_image)
    %time clip.write_videofile(output, audio=False)





def main():
  ystart = 400
  ystop = 656
  scale = 1.5

  ### TRAINING #####
  #train_model(cars, notcars)
  
  ### INFERENCE #####   
  #myimage = mpimg.imread('./test1.jpg')
  myvid = 'project_video.mp4' 
  find_vehicles_in_video(myvid)
  #new_img =find_vehicles_in_frame(myimage)
  #plt.imshow(new_img)
  #out_img, box_list = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
  
  
  #heat = np.zeros_like(image[:,:,0]).astype(np.float)

  # Add heat to each box in box list
  #heat = add_heat(heat,box_list)
      
  # Apply threshold to help remove false positives
  #heat = apply_threshold(heat,1)

  # Visualize the heatmap when displaying    
  #heatmap = np.clip(heat, 0, 255)

  # Find final boxes from heatmap using label function
  #labels = label(heatmap)
  #draw_img = draw_labeled_bboxes(np.copy(image), labels)

  #fig = plt.figure()
  #plt.subplot(121)
  #plt.imshow(image)
  #plt.title('Original')
  #plt.subplot(121)
  #plt.imshow(draw_img)
  #plt.title('Car Positions')
  #plt.subplot(122)
  #plt.imshow(heatmap, cmap='hot')
  #plt.title('Heat Map')
  #fig.tight_layout()

if __name__ == '__main__':
    main()




