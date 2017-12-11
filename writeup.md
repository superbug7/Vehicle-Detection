## Vehicle Detection
### Hot virtual pursuit of all cars on the road. 

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

*Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
*Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
*Normalize features and randomize a selection for training and testing.
*Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
*Run your pipeline on a video stream


[//]: # (Image References)
[image2]: ./images/image-2.JPG
[image3]: ./images/image-3.JPG
[image4]: ./images/image-4.JPG
[image5]: ./images/image-5.JPG
[image6]: ./images/image-6.JPG
[image7]: ./images/image-7.JPG
[video1]: ./tracked_project_video.mp4


### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Data selection

The most important aspect for training y Linear SVM classifier is the balanced data set with one side being vehicle data se having images of cars from different angles. For this I merged KITTI and GTI dataset after changing filenames and created one folder. Equally important other side of the balance is non-vehicle dataset, meaning anything on road that is not a vehicle; trees etc. This is very much need to make the classifier smart in understanding what is a car and what is not, so that it rightly detects what it is supposed to and keep everyone safe. I had around 9000 images in both 'cars' list and 'notcars' list; extracted in line 27-35 in pipeline.py

### Features (Color vect + Histogram + HOG)

#### 1. Explain how (and identify where in your code) you extracted features from the training images.

The code for this step is contained in lines 56-69 of pipeline.py. The function 'extract_features' is defined in helper_func.py file. 

*I used all three types of features and concatenated them: spatial, histogram and HOG
* I run extract_features on both vehicles and non-vehicles dataset and then concatenated all three type of features
* I tried running Grid Search and I did get slightly better accuracy, but dropped the idea due to lots of computation cycles spent with minimal return for this case. 

I then explored different color spaces and different parameters 

Here is how images look in RGB and YCrCb color space. I tried HLS and LUV and eventually selected YCrCb. 

[image2]
rgb

[image3]
YCrCb


#### 2. Explain how you settled on your final choice of HOG parameters.

After lot of tuning during exercises in Udacity course, I selected following values for these parameters:

'''
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()
'''

[image4]



[image5]



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using randomized train set after splitting some data for validataion. Line 81-82 splits and randomizes dataset into train and validation. 

Line 89-96 is where i fit the LinearSVC model on training data and then ceck the accuracy on test set. 

I got 99.08% accuracy for the dataset that I have used.  I did try GridSearch but dropped it due to a highly increased parameter tuning time with minimal improvment in accuracy 


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

Line 111 in helper_func.py shows slide_window function. I used the overlap factor 0.5 and y_start_stop[0] value of 450 so that I am looking at bottom half portion and not trying to find cars in air. 

#### 2. What did you do to optimize the performance of your classifier?

* YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector provided best result.
* Used heat maps to be properly sure about detecting right vehicles on the road. 
* Careful selection of HOG parameters 

[image7]



[image6]


---

### Video Implementation

#### 1. Provide a link to your final video output.  

Here's the video: [video1](./tracked_project_video.mp4)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

My pipeline needs to be more robust to detect some vehicles which may be further than the scope I am checking. Also, another thing to make it more robust is to see how it performs when there is more clutter of cars in left and right of my car. 

