**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example1.png
[image3]: ./examples/HOG_example2.png
[image4]: ./examples/sliding_windows.png
[image5]: ./examples/results.png
[image6]: ./examples/bboxes_heat_result.png
[video1]: ./project_video_annotated.mp4


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images. Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Vehicles: 8792
Non-vehicles: 8968

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `GRAY` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][image2]
![alt text][image3]

#### 2. Explain how you settled on your final choice of HOG parameters.

I did several experiments and decided to choose this parameters listed before and in 5th code cell. I choose GRAY color space and my idea was detect cars by only structure information not by color because car can have a lot of different colors. Also when we reduce number of features classifier warks faster.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I decided to use SVM classifier (code cell 8) with rbf kernel and default sklearn parameters. It works quite fast and shows accuracy more than 98%

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

I measured sizes of vehicles and decided to search (code cell 9) in bottom half of the image in 3 different scales:
128x128
96x96
72x72

![alt text][image4]

#### 2. Show some examples of test images to demonstrate how your pipeline is working. What did you do to optimize the performance of your classifier?

I plaed with window sizes and HOG features to ptimize performance and accuracy of the detection, after these experiments I got a nice result. Here are some example images:

![alt text][image5]

---

### Video Implementation

#### 1. Provide a link to your final video output. Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_annotated.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video. From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. After that I created average box. Also I used boxes from ten last frames.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project. Where will your pipeline likely fail?  What could you do to make it more robust?

Mostly my pipeline works well. It may fail in when two vehicle are close to each other. In that case my pipeline creates only one box instead of two. We can fix it by tune sliding windows size and overlapping. Also it missed car on several frames in the middle of the video. I hope we can resolve this issue by adding several thousands of training images to increase accuracy of classifier.

